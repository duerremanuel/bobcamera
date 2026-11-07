from setuptools import setup

package_name = 'bob_startracker'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
    ],
    install_requires=['setuptools', 'numpy', 'scipy', 'opencv-python', 'ephem'],
    zip_safe=True,
    maintainer='Bob Universal Object Tracker',
    maintainer_email='uapbob@gmail.com',
    description='Star-tracker calibration: fisheye pixels -> true-north azimuth / elevation',
    license='GPLv3',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'startracker = bob_startracker.startracker_node:main',
            'bob_startracker = bob_startracker.cli:main',
        ],
    },
)
