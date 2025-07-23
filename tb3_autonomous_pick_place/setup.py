from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'tb3_autonomous_pick_place'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
                ('share/' + package_name + '/models', glob('models/*')), 
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),  
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='anglight',
    maintainer_email='anglight@todo.todo',
    description='Autonomous pipeline for slam based mapping navigation, object detction and pick and place for turtlebot3',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'frontier_node = tb3_autonomous_pick_place.frontier_node:main',
            'detector_node = tb3_autonomous_pick_place.yolov8_detector_node:main',
            'fusion_node = tb3_autonomous_pick_place.sensor_fusion_node:main',
            'ik_node = tb3_autonomous_pick_place.compute_ik_node:main',
            'move_node = tb3_autonomous_pick_place.move_to_pose:main',
            
        ],
    },
)
