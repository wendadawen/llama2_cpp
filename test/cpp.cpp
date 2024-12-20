#include <bits/stdc++.h>
using namespace std;
struct Config {
    uint32_t magic_number;  // has to be 0x616b3432, i.e. "ak42" in ASCII
    int version; // 2
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads
    int vocab_size; // vocabulary size
    int seq_len; // max sequence length
    uint8_t shared_classifier;  // whether to share classifier weights with token embedding table
    int group_size;  // quantization group size
    void init(std::fstream &file) {
        // file.read(reinterpret_cast<char*>(this), sizeof(Config));
        // GS = group_size;
        #ifdef DEBUG
        std::cerr << "Config:"
            << "\n\tmagic_number=" << magic_number
            << "\n\tversion=" << version
            << "\n\tdim=" << dim
            << "\n\thidden_dim=" << hidden_dim
            << "\n\tn_layers=" << n_layers
            << "\n\tn_heads=" << n_heads
            << "\n\tn_kv_heads=" << n_kv_heads
            << "\n\tvocab_size=" << vocab_size
            << "\n\tseq_len=" << seq_len
            << "\n\tshared_classifier=" << shared_classifier
            << "\n\tgroup_size=" << group_size << std::endl;
        #endif
    }
};

int main() {
    Config config;
    cout << sizeof(config) << endl;
    cout << sizeof(uint32_t) + sizeof(int) + 7 * sizeof(int) + sizeof(uint8_t) + sizeof(int) << endl;
    cout << sizeof(uint8_t) << endl;
    return 0;
}