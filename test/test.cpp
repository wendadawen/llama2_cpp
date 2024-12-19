#include <bits/stdc++.h>
using namespace std;

int main() {
    // vector<vector<int>> a(3, vector<int>(4, 0));
    // for(int i = 0; i < 3; i ++) {
    //     for(int j = 0; j < 4; j ++) {
    //         cout << a[i][j] << " ";
    //     }
    //     cout << endl;
    // }

    // 测试memset和fill的性能：memset更快memset: 1761ms，fill: 7143ms
    // vector<int> a(10000000);
    // clock_t start = clock();
    // for(int i = 0; i < 1000; i ++) {
    //     memset(a.data(), 0, a.size() * sizeof(int));
    // }
    // cout << "memset: " << (clock() - start) * 1000.0 / CLOCKS_PER_SEC << "ms" << endl;

    // start = clock();
    // for(int i = 0; i < 1000; i ++) {
    //     fill(a.begin(), a.end(), 0);
    // }
    // cout << "fill: " << (clock() - start) * 1000.0 / CLOCKS_PER_SEC << "ms" << endl;
    return 0;
}