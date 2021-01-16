import numpy as np

N = 1024;
c = np.empty((N, N))
b = np.empty((N, N))
s1 = np.array(N)
s2 = np.array(N);
len1 = 0
len2 = 0;


def LCS():
    for i in range(1, len1):
        for j in range(1, len2):
            # 注：此处的s1与s2序列是从s1[0]与s2[0]开始的
            if (s1[i - 1] == s2[j - 1]):
                c[i][j] = c[i - 1][j - 1] + 1;
                b[i][j] = 1;
            else:
                if (c[i][j - 1] >= c[i - 1][j]):
                    c[i][j] = c[i][j - 1]
                    b[i][j] = 2
                else:
                    c[i][j] = c[i - 1][j]
                    b[i][j] = 3


def LCS_PRINT(i, j):
    if (i == 0 or j == 0):
        return;
    if (b[i][j] == 1):
        LCS_PRINT(i - 1, j - 1);
        print(s1[i - 1])

    else:
        if (b[i][j] == 2):
            LCS_PRINT(i, j - 1);

        else:
            LCS_PRINT(i - 1, j);


if __name__ == '__main__':
    print("请输入X字符串")
    s1 = input();
    print("请输入Y字符串")
    s2 = input();
    len1 = len(s1)
    len2 = len(s2)
    for i in range(0, len1):
        c[i][0] = 0;
    for j in range(0, len2):
        c[0][j] = 0;
    LCS();
    print("s1与s2的最长公共子序列的长度是：%d") % c[len1][len2]
    print("s1与s2的最长公共子序列是：%d") % LCS_PRINT(len1, len2)
    # return 0;
