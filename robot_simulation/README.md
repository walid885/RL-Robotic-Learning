# Robot Simulation System

A modular PyBullet-based robot simulation system for humanoid robots with advanced stabilization and motion control.

## Project Structure

```
robot_simulation/
├── src/
│   ├── core/           # Core robot components
│   ├── physics/        # Physics engine management
│   ├── control/        # Control systems
│   ├── motion/         # Motion planning
│   └── utils/          # Utilities and data structures
├── config/             # Configuration files
├── tests/              # Unit and integration tests
├── docs/               # Documentation
└── examples/           # Example scripts
```

## Installation

```bash
cd robot_simulation
pip install -e .
```

## Usage

```bash
python src/main.py
```

## Modules

### Core (`src/core/`)
- **Robot**: Main robot class handling loading and basic operations
- **JointManager**: Joint discovery, categorization, and information management
- **SimulationState**: State tracking and monitoring

### Physics (`src/physics/`)
- **PhysicsEngine**: PyBullet engine initialization and configuration
- **Environment**: Ground plane and environmental setup

### Control (`src/control/`)
- **RobotStabilizer**: Multi-phase stabilization system
- **JointController**: Low-level joint control
- **BalanceController**: Balance maintenance and recovery

### Motion (`src/motion/`)
- **WaveMotion**: Waving motion implementation
- **PoseCalculator**: Optimal pose calculation
- **MotionExecutor**: Motion execution and coordination

### Utils (`src/utils/`)
- **DataStructures**: Configuration and data classes
- **StabilityMonitor**: Robot stability monitoring
- **SimulationLogger**: Logging and debugging utilities

## Configuration

Edit `config/simulation_config.yaml` to modify simulation parameters.

## Development

Run tests:
```bash
python -m pytest tests/
```

## Features

- Modular architecture for easy extension
- Advanced stabilization system
- Configurable motion parameters
- Comprehensive monitoring
- YAML-based configuration
