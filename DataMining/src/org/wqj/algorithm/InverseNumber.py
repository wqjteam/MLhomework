# 输出逆序数
import numpy as np

# 逆序数
class Solution:
    # 空对数
    def InversePairs(self, array):
        if (array is None or len(array) == 0):
            return 0;
        newArray = np.empty((1, len(array))).flatten()
        # 复制
        for i in range(len(array)):
            newArray[i] = array[i]
        count = self.InversePairsRecursion(array, newArray, 0, len(array) - 1);
        return count;

    def InversePairsRecursion(self, data, newArray, start, end):
        if start == end:
            newArray[start] = data[start]
            return 0
        mid = (end + start)//2
        # 进行递归，细化导最小
        left = self.InversePairsRecursion(newArray, data, start, mid)
        right = self.InversePairsRecursion(newArray, data, mid + 1, end)
        # i初始化为前半段最后一个数字的下标
        i = mid
        # j初始化为后半段最后一个数字的下标
        j = end

        index_c = end
        count = 0
        while i >= start and j >= mid + 1:
            if data[i] > data[j]:
                print("逆序数对为：%d  %d"%(data[i] ,data[j] ))
                newArray[index_c] = data[i]
                index_c -= 1
                i -= 1
                count += j - mid
            else:
                newArray[index_c] = data[j]
                index_c -= 1
                j -= 1

        while i >= start:
            newArray[index_c] = data[i]
            index_c -= 1
            i -= 1
        while j >= mid + 1:
            newArray[index_c] = data[j]
            index_c -= 1
            j -= 1
        return left + right + count



s = Solution()
print("逆序数对为：11  -14")
print("逆序数对为：0  -14")
array = [11, 0, -14, -7, 17, -2, 16, 22]
print("总个数为：%d"%s.InversePairs(array))


# 使用数据的index进行求解
def InversePairs2(self, data):
    pairarray = np.empty(1, len(data)).flatten()
    if len(data) <= 0:
        return 0
    count = 0
    copy = []
    for i in range(len(data)):
        copy.append(data[i])
    copy.sort()
    i = 0
    while len(copy) > i:
        count += data.index(copy[i])
        data.remove(copy[i])
        i += 1
    return count


