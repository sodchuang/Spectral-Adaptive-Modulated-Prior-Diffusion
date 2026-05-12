class BaseFlowMatcher:
    """Abstract base class for flow matchers."""

    def compute_loss(self, model, target, **kwargs):
        """Compute training loss.

        Args:
            model: The flow network (callable).
            target: Target tensor x_1.
            **kwargs: Additional keyword arguments forwarded to the model.

        Returns:
            Tuple of (loss tensor, dict of metrics).
        """
        raise NotImplementedError

    def sample(self, model, shape, device, num_steps, return_traces=False, **kwargs):
        """Generate samples by integrating the learned velocity field.

        Args:
            model: The flow network.
            shape: Output shape (batch_size, ...).
            device: Torch device.
            num_steps: Number of Euler integration steps.
            return_traces: If True, also return trajectory and velocity histories.
            **kwargs: Additional keyword arguments forwarded to the model.

        Returns:
            Sampled tensor, or (tensor, (traj_history, vel_history)) when
            return_traces is True.
        """
        raise NotImplementedError
