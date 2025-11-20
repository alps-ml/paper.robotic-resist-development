import sys
from setuptools import setup
import os
from glob import glob
package_name = 'meca_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jessicamyers',
    maintainer_email='myersjm@rose-hulman.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={ # package name, file name, func
        'console_scripts': [
            "meca_driver = meca_controller.meca_driver:main",
            "meca_driver_test = meca_controller.meca_driver_test:main",
            "meca_control = meca_controller.meca_control:main",
            "meca_calibration = meca_controller.meca_calibration:main",
            "meca_gotoPose = meca_controller.meca_gotoPose:main",
            "meca_demo_develop_20250624 = meca_controller.meca_demo_develop_20250624:main",
            "meca_test_stirring = meca_controller.meca_test_stirring:main",
            "meca_demo_develop = meca_controller.meca_demo_develop:main",
        ],
    },
)
