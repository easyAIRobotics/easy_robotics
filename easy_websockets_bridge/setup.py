from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'easy_websockets_bridge'

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

        # # install config files
        # (os.path.join('share', package_name, 'config'),
        #     glob('config/*')),

    ],
    install_requires=[
        'setuptools',
        'websockets',
    ],
    zip_safe=True,
    maintainer='Hoang Dung Dinh',
    maintainer_email='dinhhoangdung0712@gmail.com',
    description='ROS2-websockets bridge for Web UI',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'websockets_bridge_node = easy_websockets_bridge.websockets_bridge_node:main'
        ],
    },
)
