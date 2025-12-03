from setuptools import setup

package_name = 'cse180_warehouse_project'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='CSE180 Team',
    maintainer_email='team@example.com',
    description='Warehouse robot controller',
    license='MIT',
    entry_points={
        'console_scripts': [
            'warehouse_controller = cse180_warehouse_project.warehouse_controller:main',
        ],
    },
)
