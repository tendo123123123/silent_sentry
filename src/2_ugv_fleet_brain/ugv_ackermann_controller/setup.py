from setuptools import find_packages, setup

package_name = 'ugv_ackermann_controller'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/joystick_teleop.launch.py',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aditya-pachauri',
    maintainer_email='adi.pachauri.444@gmail.com',
    description='Ackermann UGV twist controller and teleop for desert environments',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ackermann_twist_controller = ugv_ackermann_controller.ackermann_twist_controller:main',
        ],
    },
)
