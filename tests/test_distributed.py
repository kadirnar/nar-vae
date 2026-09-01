"""Network-free tests for torchrun environment and device contracts."""

import unittest
from unittest.mock import Mock, call, patch

import torch

from nar_vae.distributed import (
    DistributedContext,
    distributed_cleanup_guard,
    initialize_distributed,
    propagate_distributed_error,
    propagate_process_group_error,
    require_process_group_consistent_value,
    resolve_node_consistent_value,
    run_distributed_operation,
    shard_indices,
)


class DistributedContextTest(unittest.TestCase):
    def test_single_process_defaults_do_not_claim_ddp(self):
        context = DistributedContext.from_environment({})

        self.assertFalse(context.is_distributed)
        self.assertTrue(context.is_main_process)
        self.assertEqual(context.trainer_local_rank, -1)

    def test_torchrun_ranks_keep_global_and_local_identity_separate(self):
        context = DistributedContext.from_environment(
            {
                "LOCAL_RANK": "2",
                "RANK": "10",
                "WORLD_SIZE": "16",
                "LOCAL_WORLD_SIZE": "8",
            }
        )

        self.assertTrue(context.is_distributed)
        self.assertFalse(context.is_main_process)
        self.assertEqual(context.local_rank, 2)
        self.assertEqual(context.rank, 10)
        self.assertEqual(context.trainer_local_rank, 2)
        self.assertEqual(context.device(), torch.device("cuda:2"))

    def test_distributed_launch_requires_complete_rank_metadata(self):
        with self.assertRaisesRegex(ValueError, "LOCAL_RANK is required"):
            DistributedContext.from_environment({"WORLD_SIZE": "8", "RANK": "0"})
        with self.assertRaisesRegex(ValueError, "smaller than WORLD_SIZE"):
            DistributedContext.from_environment({"WORLD_SIZE": "2", "LOCAL_RANK": "0", "RANK": "2"})

    def test_cuda_device_is_selected_before_process_group_initialization(self):
        context = DistributedContext(local_rank=1, rank=1, world_size=2, local_world_size=2)
        events = Mock()

        with (
            patch("nar_vae.distributed.torch.cuda.is_available", return_value=True),
            patch("nar_vae.distributed.torch.cuda.device_count", return_value=2),
            patch(
                "nar_vae.distributed.torch.cuda.set_device",
                side_effect=lambda rank: events("set_device", rank),
            ),
            patch("nar_vae.distributed.dist.is_initialized", return_value=False),
            patch(
                "nar_vae.distributed.dist.init_process_group",
                side_effect=lambda **kwargs: events("init", kwargs["backend"]),
            ),
            patch("nar_vae.distributed.dist.get_rank", return_value=1),
            patch("nar_vae.distributed.dist.get_world_size", return_value=2),
        ):
            result = initialize_distributed(context)

        self.assertIs(result, context)
        self.assertEqual(
            events.call_args_list,
            [call("set_device", 1), call("init", "nccl")],
        )

    def test_rank_outside_visible_cuda_devices_fails_before_initialization(self):
        context = DistributedContext(local_rank=2, rank=2, world_size=4, local_world_size=4)
        with (
            patch("nar_vae.distributed.torch.cuda.is_available", return_value=True),
            patch("nar_vae.distributed.torch.cuda.device_count", return_value=2),
            patch("nar_vae.distributed.dist.init_process_group") as initialize,
            self.assertRaisesRegex(RuntimeError, "cannot select"),
        ):
            initialize_distributed(context)

        initialize.assert_not_called()

    def test_strided_data_shards_are_disjoint_and_complete(self):
        shards = [set(shard_indices(11, rank=rank, world_size=4)) for rank in range(4)]

        self.assertEqual(set.union(*shards), set(range(11)))
        for rank, shard in enumerate(shards):
            for other_rank, other in enumerate(shards):
                if rank != other_rank:
                    self.assertFalse(shard.intersection(other))

        with self.assertRaisesRegex(ValueError, "0 <= rank"):
            shard_indices(5, rank=2, world_size=2)

    def test_new_process_group_cleanup_avoids_barrier_after_failure(self):
        with (
            patch(
                "nar_vae.distributed.dist.is_initialized",
                side_effect=[False, True],
            ),
            patch("nar_vae.distributed.cleanup_distributed") as cleanup,
            self.assertRaisesRegex(RuntimeError, "rank-local failure"),
        ):
            with distributed_cleanup_guard():
                raise RuntimeError("rank-local failure")

        cleanup.assert_called_once_with(barrier=False)

    def test_new_process_group_cleanup_synchronizes_after_success(self):
        with (
            patch(
                "nar_vae.distributed.dist.is_initialized",
                side_effect=[False, True],
            ),
            patch("nar_vae.distributed.cleanup_distributed") as cleanup,
            distributed_cleanup_guard(),
        ):
            pass

        cleanup.assert_called_once_with(barrier=True)

    def test_preexisting_process_group_is_not_owned_by_guard(self):
        with (
            patch("nar_vae.distributed.dist.is_initialized", return_value=True),
            patch("nar_vae.distributed.cleanup_distributed") as cleanup,
            distributed_cleanup_guard(),
        ):
            pass

        cleanup.assert_not_called()

    def test_node_leaders_must_resolve_identical_values(self):
        context = DistributedContext(local_rank=0, rank=2, world_size=4, local_world_size=2)

        def identical(outputs, local_payload):
            outputs[:] = [
                {"rank": 0, "is_node_leader": True, "value": {"hash": "same"}, "error": None},
                {"rank": 1, "is_node_leader": False, "value": None, "error": None},
                local_payload,
                {"rank": 3, "is_node_leader": False, "value": None, "error": None},
            ]

        with patch("nar_vae.distributed.dist.all_gather_object", side_effect=identical):
            value = resolve_node_consistent_value(
                context,
                lambda: {"hash": "same"},
                description="dataset identity",
            )
        self.assertEqual(value, {"hash": "same"})

        def mismatched(outputs, local_payload):
            outputs[:] = [
                {"rank": 0, "is_node_leader": True, "value": {"hash": "other"}, "error": None},
                {"rank": 1, "is_node_leader": False, "value": None, "error": None},
                local_payload,
                {"rank": 3, "is_node_leader": False, "value": None, "error": None},
            ]

        with (
            patch("nar_vae.distributed.dist.all_gather_object", side_effect=mismatched),
            self.assertRaisesRegex(ValueError, "differs between node-local"),
        ):
            resolve_node_consistent_value(
                context,
                lambda: {"hash": "same"},
                description="dataset identity",
            )

    def test_rank_local_errors_are_propagated_before_later_collectives(self):
        context = DistributedContext(local_rank=1, rank=1, world_size=2, local_world_size=2)

        def remote_failure(outputs, local_message):
            self.assertIsNone(local_message)
            outputs[:] = ["OSError: disk full", None]

        with (
            patch("nar_vae.distributed.dist.all_gather_object", side_effect=remote_failure),
            self.assertRaisesRegex(RuntimeError, "Rank 0 failed during final export"),
        ):
            propagate_distributed_error(context, None, description="final export")

    def test_live_process_group_propagates_remote_checkpoint_failure(self):
        def remote_failure(outputs, local_message):
            self.assertIsNone(local_message)
            outputs[:] = [None, "OSError: disk full"]

        with (
            patch("nar_vae.distributed.dist.is_available", return_value=True),
            patch("nar_vae.distributed.dist.is_initialized", return_value=True),
            patch("nar_vae.distributed.dist.get_world_size", return_value=2),
            patch("nar_vae.distributed.dist.all_gather_object", side_effect=remote_failure),
            self.assertRaisesRegex(
                RuntimeError,
                "Rank 1 failed during Trainer checkpoint save",
            ),
        ):
            propagate_process_group_error(None, description="Trainer checkpoint save")

    def test_live_process_group_preserves_local_checkpoint_exception(self):
        failure = OSError("disk full")

        def local_failure(outputs, local_message):
            self.assertEqual(local_message, "OSError: disk full")
            outputs[:] = [local_message, None]

        with (
            patch("nar_vae.distributed.dist.is_available", return_value=True),
            patch("nar_vae.distributed.dist.is_initialized", return_value=True),
            patch("nar_vae.distributed.dist.get_world_size", return_value=2),
            patch("nar_vae.distributed.dist.all_gather_object", side_effect=local_failure),
            self.assertRaises(OSError) as raised,
        ):
            propagate_process_group_error(failure, description="Trainer checkpoint save")

        self.assertIs(raised.exception, failure)

    def test_distributed_operation_propagates_before_returning(self):
        context = DistributedContext(local_rank=0, rank=0, world_size=1, local_world_size=1)

        self.assertEqual(
            run_distributed_operation(context, lambda: "ready", description="model setup"),
            "ready",
        )
        with self.assertRaisesRegex(OSError, "corrupt parent"):
            run_distributed_operation(
                context,
                lambda: (_ for _ in ()).throw(OSError("corrupt parent")),
                description="model setup",
            )

    def test_live_process_group_requires_identical_resume_identity(self):
        def mismatched(outputs, local_value):
            outputs[:] = [local_value, {"sha256": "different"}]

        with (
            patch("nar_vae.distributed.dist.is_available", return_value=True),
            patch("nar_vae.distributed.dist.is_initialized", return_value=True),
            patch("nar_vae.distributed.dist.get_world_size", return_value=2),
            patch("nar_vae.distributed.dist.all_gather_object", side_effect=mismatched),
            self.assertRaisesRegex(ValueError, "differs between distributed ranks"),
        ):
            require_process_group_consistent_value(
                {"sha256": "local"},
                description="resume identity",
            )


if __name__ == "__main__":
    unittest.main()
