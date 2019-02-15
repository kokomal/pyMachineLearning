# coding:utf-8
'''
Created on 2019-02-15
Using Python 3.6.3
@author: chenyuanjun
'''
S={'A','B','C'}

def move(D, frm, to):
    print('Moving Disk of Size %d from %s to %s' % (D, frm, to))

def hannoi(D, frm, to):
    if (D == 1):
        move(D, frm, to)
        return
    inner = (list(S - {frm} - {to}))[0]
    hannoi(D - 1, frm, inner)
    move(D, frm, to)
    hannoi(D - 1, inner, to)
    
if __name__=='__main__':
    hannoi(6, 'A', 'C')