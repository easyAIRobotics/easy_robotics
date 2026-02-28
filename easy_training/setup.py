from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'easy_training'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # ROS2 index
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),

        # package.xml
        ('share/' + package_name, ['package.xml']),

        # install launch files
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),

        # install config files
        (os.path.join('share', package_name, 'config'),
            glob('config/*')),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Hoang Dung Dinh',
    maintainer_email='dinhhoangdung0712@gmail.com',
    description='Robotics agent training',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'demo_rl_trainer = easy_training.demo_rl_trainer:main',
            'experts_node = easy_training.experts_node:main',
        ],
    },
)
