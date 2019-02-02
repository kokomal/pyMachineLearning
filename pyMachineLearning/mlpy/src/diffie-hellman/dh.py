# coding:utf-8
'''
Created on 2019-02-02
Using Python 3.6.3
@author: chenyuanjun
'''
import time


def enc(base, exp, modulus):
    return base ** (exp) % modulus


def crack(gpPair, A, B, stopAtOne=True):
    print('Now a hacker wants to crack the code\nHe/she has the information: gpPair=', gpPair, ', and A =', A, ', B =', B)
    limit = max(A,B)
    print('Assume the random a, b is within %d...' % limit)
    time_start = time.time()
    for i in range(1, limit):
        for j in range(1, limit):
            tryA = enc(gpPair[0], i, gpPair[1])
            tryB = enc(gpPair[0], j, gpPair[1])
            if (A == tryA and B == tryB):
                codeA = enc(B, i, gpPair[1])
                codeB = enc(A, j, gpPair[1])
                if (codeA == codeB):
                    print("CRACKED!", codeA, codeB, 'and candidate a =', i, ',and b =', j)
                    if stopAtOne:
                        time_end = time.time()
                        print('Totally Time Cost', time_end - time_start, 's')
                        return  
    print("END WITHOUT CRACKED!")


class Person:

    def __init__(self, priv):
        self.aorb = priv

    def genAorB(self, gpPair):
        return enc(gpPair[0], self.aorb, gpPair[1])

    def genPrivCode(self, gpPair, AorB):
        return enc(AorB, self.aorb, gpPair[1])


def diffie_hellman_demo2(gpPair, a, b):
    alice = Person(a)
    bob = Person(b)
    
    A = alice.genAorB(gpPair) # A,B都是可以被截获的
    B = bob.genAorB(gpPair)
    
    # swap，即Alice与Bob交换A和B
    privCodeAlice = alice.genPrivCode(gpPair, B)
    privCodeBob = bob.genPrivCode(gpPair, A)
    print('Alice has her private code %s, while bob has his private code %s' % (privCodeAlice, privCodeBob))
    crack(gpPair, A, B)


if __name__ == '__main__':    
    gpPair = (733, 1519)  # 公钥对任何人开放，包括黑客, g代表底数，p代表模，g和p都是素数
    print('G-P Pair is:', gpPair)
    
    a = 1237  # 双方各自私自生成，不公开，这个是指数，因此不应该太大
    b = 2347  # 
    
    diffie_hellman_demo2(gpPair, a, b)

