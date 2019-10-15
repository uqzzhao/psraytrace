'''
@author:     Zhengguang Zhao
@copyright:  Copyright 2016-2019, Zhengguang Zhao.
@license:    MIT
@contact:    zg.zhao@outlook.com

'''

from examples import example2d, example3d, example_vert_hor

def main():

    example2d.run() # a downhole monitoring example
    example3d.run() # a surface monitoring example
    #example_vert_hor.run() # src and rcv pairs for shooting vertically and horizontally


# This will actually run the code if called stand-alone:
if __name__ == '__main__':
    main()