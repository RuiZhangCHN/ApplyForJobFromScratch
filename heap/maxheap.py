class MaxHeap(object):

    def __init__(self):
        self._data = []

    def size(self):
        return len(self._data)

    def is_empty(self):
        return len(self._data) == 0

    def add(self, item):
        self._data.append(item)
        self._shiftup(len(self._data)-1)

    def pop(self):
        if not self.is_empty():
            self._data[0], self._data[-1] = self._data[-1], self._data[0]
            ret = self._data.pop()
            self._shiftdown(0)

            return ret

    def _shiftup(self, index):
        parent_idx = (index-1) >> 1
        while index > 0 and self._data[index] > self._data[parent_idx]:
            self._data[index], self._data[parent_idx] = self._data[parent_idx], self._data[index]
            index = parent_idx
            parent_idx = (index-1) >> 1

    def _shiftdown(self, index):
        j = (index << 1) + 1
        while j < self.size():
            if j+1 < self.size() and self._data[j+1] > self._data[j]:
                j += 1
            if self._data[index] > self._data[j]:
                break
            self._data[index], self._data[j] = self._data[j], self._data[index]
            index = j
            j = (index << 1) + 1

import random

def testIntValue():
    for iTimes in range(10):
        iLen = random.randint(1, 300)
        allData = random.sample(range(iLen * 100), iLen)
        print('\nlen =', iLen)

        oMaxHeap = MaxHeap()
        print('_data:\t   ', allData)
        arrDataSorted = sorted(allData, reverse=True)
        print('dataSorted:', arrDataSorted)
        for i in allData:
            oMaxHeap.add(i)
        heapData = []
        for i in range(iLen):
            iExpected = arrDataSorted[i]
            iActual = oMaxHeap.pop()
            heapData.append(iActual)
            print('{0}, expected: {1}, actual: {2}'.format(iExpected == iActual, iExpected, iActual))
            assert iExpected == iActual, ""
        print('dataSorted:', arrDataSorted)
        print('heapData:  ', heapData)

def testTupleValue():
    for iTimes in range(10):
        iLen = random.randint(1, 300)
        listData = random.sample(range(iLen * 100), iLen)
        #         listData = [1, 4, 3, 2, 5, 7, 6]
        #         iLen = len(listData)
        # 注意：key作为比较大小的关键
        allData = dict(zip(listData, [str(e) for e in listData]))
        print('\nlen =', iLen)
        print('allData: ', allData)

        oMaxHeap = MaxHeap()
        arrDataSorted = sorted(allData.items(), key=lambda d: d[0], reverse=True)
        #         arrDataSorted = sorted(allData, reverse=True)
        print('dataSorted:', arrDataSorted)
        for (k, v) in allData.items():
            oMaxHeap.add((k, v))  # 元祖的第一个元素作为比较点
        heapData = []
        for i in range(iLen):
            iExpected = arrDataSorted[i]
            iActual = oMaxHeap.pop()
            heapData.append(iActual)
            print('{0}, expected: {1}, actual: {2}'.format(iExpected == iActual, iExpected, iActual))
            assert iExpected == iActual, ""
        print('dataSorted:', arrDataSorted)
        print('heapData:  ', heapData)


def testClassValue():
    class Model4Test(object):
        '''
        用于放入到堆的自定义类。注意要重写__lt__、__ge__、__le__和__cmp__函数。
        '''

        def __init__(self, sUid, value):
            self._sUid = sUid
            self._value = value

        def getUid(self):
            return self._sUid

        def getValue(self):
            return self._value

        # 类类型，使用的是小于号_lt_
        def __lt__(self, other):  # operator <
            #             print('in __lt__(self, other)')
            return self.getValue() < other.getValue()

        def __ge__(self, other):  # oprator >=
            return self.getValue() >= other.getValue()

        # 下面两个方法重写一个就可以了
        def __le__(self, other):  # oprator <=
            return self.getValue() <= other.getValue()

        def __cmp__(self, other):
            if self.getValue() < other.getValue():
                return -1
            if self.getValue() > other.getValue():
                return 1
            return 0

        def __str__(self):
            return '({0}, {1})'.format(self._value, self._sUid)

    for iTimes in range(10):
        iLen = random.randint(1, 300)
        listData = random.sample(range(iLen * 100), iLen)
        #         listData = [1, 4, 3, 2, 5, 7, 6]
        allData = [Model4Test(str(value), value) for value in listData]
        print('allData:   ', [str(e) for e in allData])
        iLen = len(allData)
        print('\nlen =', iLen)

        oMaxHeap = MaxHeap()
        arrDataSorted = sorted(allData, reverse=True)
        print('dataSorted:', [str(e) for e in arrDataSorted])
        for i in allData:
            oMaxHeap.add(i)
        heapData = []
        for i in range(iLen):
            iExpected = arrDataSorted[i]
            iActual = oMaxHeap.pop()
            heapData.append(iActual)
            print('{0}, expected: {1}, actual: {2}'.format(iExpected == iActual, iExpected, iActual))
            assert iExpected == iActual, ""
        print('dataSorted:', [str(e) for e in arrDataSorted])
        print('heapData:  ', [str(e) for e in heapData])

testIntValue()
# testTupleValue()
# testClassValue()