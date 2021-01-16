def InversePairs(array):
    if (array is None or array.length == 0):
        return 0;
    count = InversePairs(array, 0, array.length - 1) % 1000000007;
    return count;


def InversePairs(a, low, high):
    count = 0;
    if (low < high):
        mid = (low + high) / 2;
        count += InversePairs(a, low, mid) % 1000000007;
        count += InversePairs(a, mid + 1, high) % 1000000007;
        count += merge(a, low, mid, high) % 1000000007;
    return count % 1000000007;


def merge(a, low, mid, high):
    count = 0;
    # i指向第一个有序区间的第一个元素，j指向第二有序区间的第一个元素
    i = low, j = mid + 1
    k = 0
    # 临时数组，暂存合并的有序列表
    temp = [high - low + 1];
    # 顺序比较两个区域的元素，将最小的存入临时数组中
    while (i <= mid and j <= high):
        if (a[i] <= a[j]):
            temp[k] = a[i];
            k = k + 1
            i = i + 1
        else:
            temp[k] = a[j];
            k = k + 1
            j = j + 1
    count += mid + 1 - i;
    count = count % 1000000007;
    # 第一个区域元素有剩余
    while (i <= mid):
        temp[k] = a[i]
        k += 1
        i += 1

    # 第二个区域元素有剩余
    while (j <= high):
        temp[k] = a[j];
        k += 1
        j += 1

    # 将排好序的元素，重新存回到A中
    k = 0
    for i in range(low, high + 1):
        a[i] = temp[k];
        k = k + 1
    return count % 1000000007;
