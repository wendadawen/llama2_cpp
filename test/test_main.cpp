#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cassert>

typedef std::vector<float> tensor1d;

int sample_topp(tensor1d &logits, float topp, float coin) {
    std::vector<int> ids(logits.size());
    std::iota(ids.begin(), ids.end(), 0);
    std::stable_sort(ids.begin(), ids.end(), [&](int a, int b) {
        return logits[a] > logits[b];
    });
    int last_idx = 0;
    float cumulative_prob = 0.0f;
    for(int i = 0; i < logits.size(); i ++) {
        cumulative_prob += logits[ids[i]];
        if(cumulative_prob >= topp) {
            last_idx = i;
            break;
        }
    }
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for(int i = 0; i <= last_idx; i ++) {
        cdf += logits[ids[i]];
        if(r < cdf) {
            return ids[i];
        }
    }
    return ids[last_idx];
}

void test_sample_topp() {
    {
        tensor1d logits = {0.1, 0.2, 0.3, 0.4};
        float topp = 0.5;
        float coin = 0.3;
        int result = sample_topp(logits, topp, coin);
        assert(result == 3);
    }
    {
        tensor1d logits = {0.1, 0.2, 0.3, 0.4};
        float topp = 0.8;
        float coin = 0.7;
        int result = sample_topp(logits, topp, coin);
        assert(result == 2);
    }
    {
        tensor1d logits = {0.4, 0.3, 0.2, 0.1};
        float topp = 0.7;
        float coin = 0.5;
        int result = sample_topp(logits, topp, coin);
        assert(result == 0);
    }
    {
        tensor1d logits = {0.25, 0.25, 0.25, 0.25};
        float topp = 0.5;
        float coin = 0;
        int result = sample_topp(logits, topp, coin);
        assert(result == 0);
    }

    std::cout << "All tests passed!" << std::endl;
}

int main() {
    test_sample_topp();
    return 0;
}