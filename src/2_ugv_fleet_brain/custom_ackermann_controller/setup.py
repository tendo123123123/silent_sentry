from setuptools import find_packages, setup

package_name = 'custom_ackermann_controller'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', [
            'config/enhanced_wheel_odometry.yaml',
            'config/enhanced_imu_processor.yaml',
            'config/enhanced_ekf.yaml',
            # Terramechanic localization stack configs
            'config/imu_filter.yaml',
            'config/terramechanic_odometry.yaml',
            'config/ekf.yaml',
            'config/ukf.yaml',
            'config/trn_slam.yaml',
            'config/factor_graph.yaml',
        ]),
        ('share/' + package_name + '/launch', [
            'launch/joystick_teleop.launch.py',
            'launch/enhanced_localization.launch.py',
            # Terramechanic localization launch
            'launch/terramechanic_localization.launch.py',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aditya-pachauri',
    maintainer_email='adi.pachauri.444@gmail.com',
    description='Ackermann UGV controller with terramechanics-aware localization for desert environments',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ackermann_twist_controller = custom_ackermann_controller.ackermann_twist_controller:main',
            'enhanced_wheel_odometry = custom_ackermann_controller.enhanced_wheel_odometry:main',
            'enhanced_imu_processor = custom_ackermann_controller.enhanced_imu_processor:main',
            # Terramechanic localization stack
            'terramechanic_odometry = custom_ackermann_controller.terramechanic_odometry:main',
            'local_dem_builder = custom_ackermann_controller.local_dem_builder:main',
            'trn_slam_node = custom_ackermann_controller.trn_slam_node:main',
            'odom_ground_truth_comparator = custom_ackermann_controller.odom_ground_truth_comparator:main',
            'odom_visualizer = custom_ackermann_controller.odom_visualizer_node:main',
            'imu_covariance_fixer = custom_ackermann_controller.imu_covariance_fixer:main',
            'factor_graph_fuser = custom_ackermann_controller.factor_graph_fuser:main',
        ],
    },
)
