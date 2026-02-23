from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'easy_perception'

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
    install_requires=[
        "setuptools",
        "numpy",
        "opencv-python",
        "aiortc",
        "av",
        "websockets",
    ],
    zip_safe=True,
    maintainer='Hoang Dung Dinh',
    maintainer_email='dinhhoangdung0712@gmail.com',
    description='Computer vision models and perception pipelines for robotics',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'human_pose_estimation_node = easy_perception.human_pose_estimation.human_pose_estimation:main',
            'object_detection_node = easy_perception.object_detection.object_detection:main'
        ],
    },
)
