from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'rl_geofencing'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='Aditya Pachauri',
    maintainer_email='adi.pachauri.444@gmail.com',
    description='Base Station RL brain for EMCON-compliant elastic geo-fencing '
                '(macroscopic fleet cohesion via a single reallocation broadcast).',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'base_station_node = rl_geofencing.base_station_node:main',
            'train = rl_geofencing.train:main',
        ],
    },
)
