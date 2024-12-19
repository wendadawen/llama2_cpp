#include <bits/stdc++.h>
using namespace std;

int main() {
    vector<float> values = {2.0, 1.0, 3.0, 4.0, 5.0};
    vector<int> ids(values.size());
    iota(ids.begin(), ids.end(), 0);
    sort(ids.begin(), ids.end(), [&](int a, int b) {
        return values[a] < values[b];
    });
    for(auto id : ids) {
        cout << id << " ";
    }
    return 0;
}