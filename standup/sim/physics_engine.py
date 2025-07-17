import pybullet as p
import pybullet_data
import time

class PhysicsEngine:
    def __init__(self, connection_mode=p.GUI):
        self.connection_mode = connection_mode
        self.client_id = -1

    def initialize(self):
        self.client_id = p.connect(self.connection_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Fixed line
        p.setGravity(0, 0, -9.81)
        if self.connection_mode == p.GUI:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

    def step_simulation(self):
        p.stepSimulation(physicsClientId=self.client_id)

    def disconnect(self):
        if self.client_id != -1:
            p.disconnect(self.client_id)
            self.client_id = -1