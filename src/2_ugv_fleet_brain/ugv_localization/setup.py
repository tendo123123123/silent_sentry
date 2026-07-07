from setuptools import find_packages, setup

package_name = 'ugv_localization'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', [
            'config/terramechanic_odometry.yaml',
            'config/trn_slam.yaml',
            'config/factor_graph.yaml',
        ]),
        ('share/' + package_name + '/launch', [
            'launch/terramechanic_localization.launch.py',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aditya-pachauri',
    maintainer_email='adi.pachauri.444@gmail.com',
    description='System-wide localization launch files, configs, and monitoring visualizer for UGV',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'odom_visualizer = ugv_localization.odom_visualizer_node:main',
        ],
    },
)
