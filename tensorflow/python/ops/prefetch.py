

from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.training import session_run_hook


@tf_export(v1=["staged"])
def staged(
        features,
        feed_list=None,
        feed_generator=None,
        capacity=1,
        num_threads=1,
        num_clients=1,
        timeout_millis=300000,
        closed_exception_types=None,
        ignored_exception_types=None,
        name=None):
    print('tf.staged fallback')
    return features


@tf_export(v1=["make_prefetch_hook"])
def make_prefetch_hook(daemon=True, start=True):
    """Create PrefetchRunner.Hook for prefetching.

    Args:
      daemon: (Optional.) Whether the threads should be marked as `daemons`,
        meaning they don't block program exit.
      start: (Optional.) If `False` threads would not be started.

    Returns:
      A PrefetchRunner.Hook for prefetching.
    """
    return session_run_hook.SessionRunHook()
