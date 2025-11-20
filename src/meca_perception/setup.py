from setuptools import find_packages, setup

package_name = 'meca_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='linqs',
    maintainer_email='fmayor@stanford.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': ['image_subscriber_node = meca_perception.image_subscriber_node:main',
                            'chip_detection_node = meca_perception.chip_detection_node:main'
        ],
    },
)
