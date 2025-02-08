import rclpy
import rclpy._rclpy_pybind11
from rclpy.client import Client
from rclpy.node import Node

from lifecycle_msgs.srv import ChangeState, GetState
from lifecycle_msgs.msg import State, Transition


class LifecycleManagerNode(Node):
    """Class to manage lifecycle node transitions."""

    # Constants
    LIFECYCLE_STATE_NAMES = {
        State.PRIMARY_STATE_UNKNOWN: "unknown",
        State.PRIMARY_STATE_UNCONFIGURED: "unconfigured",
        State.PRIMARY_STATE_INACTIVE: "inactive",
        State.PRIMARY_STATE_ACTIVE: "active",
        State.PRIMARY_STATE_FINALIZED: "finalized",
    }

    LIFECYCLE_TRANSITION_NAMES = {
        Transition.TRANSITION_CREATE: "create",
        Transition.TRANSITION_CONFIGURE: "configure",
        Transition.TRANSITION_ACTIVATE: "activate",
        Transition.TRANSITION_DEACTIVATE: "deactivate",
        Transition.TRANSITION_CLEANUP: "cleanup",
        Transition.TRANSITION_UNCONFIGURED_SHUTDOWN: "shutdown",
        Transition.TRANSITION_INACTIVE_SHUTDOWN: "shutdown",
        Transition.TRANSITION_ACTIVE_SHUTDOWN: "shutdown",
        Transition.TRANSITION_DESTROY: "destroy",
    }

    def __init__(self):
        """Lifecycle manager initializer."""
        super().__init__("lifecycle_manager")

        # Parameters
        self.declare_parameter("node_startup_order", [""])
        lifecycle_names = (
            self.get_parameter("node_startup_order").get_parameter_value().string_array_value
        )

        # Create clients to change and retrieve the states of lifecycle nodes
        self._lifecycle_clients: dict[str, dict[str, Client]] = {}

        for node_name in lifecycle_names:
            self._lifecycle_clients[node_name] = {}
            self._lifecycle_clients[node_name]["change"] = self.create_client(
                ChangeState, f"/{node_name}/change_state"
            )
            self._lifecycle_clients[node_name]["get"] = self.create_client(
                GetState, f"/{node_name}/get_state"
            )

        # Wait for the services to be available
        for node in self._lifecycle_clients.values():
            for client in node.values():
                client.wait_for_service()

        # Transition nodes to the 'inactive' state in order
        for node, client in self._lifecycle_clients.items():
            current_state_id = self._get_state(client.get("get"))

            if self._is_transition_allowed(node, current_state_id, Transition.TRANSITION_CONFIGURE):
                self._change_state(client.get("change"), Transition.TRANSITION_CONFIGURE)

        # Transition nodes to the 'active' state in order
        for node, client in self._lifecycle_clients.items():
            current_state_id = self._get_state(client.get("get"))

            if self._is_transition_allowed(node, current_state_id, Transition.TRANSITION_ACTIVATE):
                self._change_state(client.get("change"), Transition.TRANSITION_ACTIVATE)

    def _change_state(self, client: Client, transition_id: int) -> None:
        """Sends a request to change the state of a lifecycle node.

        Args:
            client: ROS 2 ChangeState client.
            transition_id: Transition identifier.

        """
        request = ChangeState.Request()
        request.transition.id = transition_id
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

    def _get_state(self, client: Client) -> int | None:
        """Sends a request to retrieve the current state of a lifecycle node.

        Args:
            client: ROS 2 GetState client.

        Returns:
            Current state identifier on success. None on failure.

        """
        request = GetState.Request()
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        return future.result().current_state.id if future.result() else None

    def _is_transition_allowed(self, node_name: str, state_id: int, transition_id: int) -> bool:
        """Checks whether a transition can be performed based on the current state of a node.

        Args:
            node_name: Node name (for logging purposes only).
            state_id: Current state identifier.
            transition_id: Desired transition identifier.

        Returns:
            True if the transition is permitted; False otherwise.

        """
        valid_transitions = {
            State.PRIMARY_STATE_UNCONFIGURED: [
                Transition.TRANSITION_CONFIGURE,
                Transition.TRANSITION_UNCONFIGURED_SHUTDOWN,
            ],
            State.PRIMARY_STATE_INACTIVE: [
                Transition.TRANSITION_ACTIVATE,
                Transition.TRANSITION_CLEANUP,
                Transition.TRANSITION_INACTIVE_SHUTDOWN,
            ],
            State.PRIMARY_STATE_ACTIVE: [
                Transition.TRANSITION_DEACTIVATE,
                Transition.TRANSITION_ACTIVE_SHUTDOWN,
            ],
            State.PRIMARY_STATE_FINALIZED: [
                Transition.TRANSITION_DESTROY,
            ],
        }

        is_allowed = transition_id in valid_transitions.get(state_id, [])

        if not is_allowed:
            state_name = LifecycleManagerNode.LIFECYCLE_STATE_NAMES.get(state_id)
            transition_name = LifecycleManagerNode.LIFECYCLE_TRANSITION_NAMES.get(transition_id)
            self.get_logger().error(
                f"The '{node_name}' node cannot execute the '{transition_name}' transition from "
                f"the '{state_name}' state."
            )

        return is_allowed


def main(args=None):
    rclpy.init(args=args)
    lifecycle_manager_node = LifecycleManagerNode()

    try:
        rclpy.spin(lifecycle_manager_node)
    except KeyboardInterrupt:
        pass

    lifecycle_manager_node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
