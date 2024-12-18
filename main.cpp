#include <bits/stdc++.h>
#define DEBUG

typedef std::vector<float> tensor1d;
typedef std::vector<tensor1d> tensor2d;
typedef std::vector<tensor2d> tensor3d;

struct Config {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads
    int vocab_size; // vocabulary size
    int seq_len; // max sequence length
    void init(std::fstream &file) {
        file.read(reinterpret_cast<char*>(this), sizeof(Config));
        #ifdef DEBUG
        std::cout << "Config:"
            << "\n\tdim=" << dim
            << "\n\thidden_dim=" << hidden_dim
            << "\n\tn_layers=" << n_layers
            << "\n\tn_heads=" << n_heads
            << "\n\tn_kv_heads=" << n_kv_heads
            << "\n\tvocab_size=" << vocab_size
            << "\n\tseq_len=" << seq_len << std::endl;
        #endif
    }
};

struct Weights {
    tensor2d token_embedding_table;  // [vocab_size, dim]
    tensor2d rms_att_weight;  // [layer, dim]
    tensor3d wq; // [layer, dim, dim]
    tensor3d wk; // [layer, dim, kv_dim], where kv_dim = dim * n_kv_heads / n_heads
    tensor3d wv; // [layer, dim, kv_dim]
    tensor3d wo; // [layer, dim, dim]
    tensor2d rms_ffn_weight;  // [layer, dim]
    tensor3d w1; // [layer, hidden_dim, dim]
    tensor3d w2; // [layer, dim, hidden_dim]
    tensor3d w3; // [layer, hidden_dim, dim]
    tensor1d rms_final_weight;  // [dim]
    tensor2d freq_cis_real;  // [seq_len, head_size/2], where head_size = dim / n_heads
    tensor2d freq_cis_imag;  // [seq_len, head_size/2]
    tensor2d wcls;  // [vocab_size, dim]
    void read_weight(tensor1d &tensor, std::fstream &file) {
        file.read(reinterpret_cast<char*>(tensor.data()), tensor.size() * sizeof(float));
    }
    void read_weight(tensor2d &tensor, std::fstream &file) {
        for (auto &t : tensor) read_weight(t, file);
    }
    void read_weight(tensor3d &tensor, std::fstream &file) {
        for (auto &t : tensor) read_weight(t, file);
    }
    void init(Config &config, std::fstream &file) {
        token_embedding_table.resize(config.vocab_size, tensor1d(config.dim));
        rms_att_weight.resize(config.n_layers, tensor1d(config.dim));
        wq.resize(config.n_layers, tensor2d(config.dim, tensor1d(config.dim)));
        wk.resize(config.n_layers, tensor2d(config.dim, tensor1d(config.dim * config.n_kv_heads / config.n_heads)));
        wv.resize(config.n_layers, tensor2d(config.dim, tensor1d(config.dim * config.n_kv_heads / config.n_heads)));
        wo.resize(config.n_layers, tensor2d(config.dim, tensor1d(config.dim)));
        rms_ffn_weight.resize(config.n_layers, tensor1d(config.dim));
        w1.resize(config.n_layers, tensor2d(config.hidden_dim, tensor1d(config.dim)));
        w2.resize(config.n_layers, tensor2d(config.dim, tensor1d(config.hidden_dim)));
        w3.resize(config.n_layers, tensor2d(config.hidden_dim, tensor1d(config.dim)));
        rms_final_weight.resize(config.dim);
        freq_cis_real.resize(config.seq_len, tensor1d(config.dim / config.n_heads / 2));
        freq_cis_imag.resize(config.seq_len, tensor1d(config.dim / config.n_heads / 2));
        wcls.resize(config.vocab_size, tensor1d(config.dim));

        read_weight(token_embedding_table, file);
        read_weight(rms_att_weight, file);
        read_weight(wq, file);
        read_weight(wk, file);
        read_weight(wv, file);
        read_weight(wo, file);
        read_weight(rms_ffn_weight, file);
        read_weight(w1, file);
        read_weight(w2, file);
        read_weight(w3, file);
        read_weight(rms_final_weight, file);
        read_weight(freq_cis_real, file);
        read_weight(freq_cis_imag, file);
        if(config.vocab_size < 0) read_weight(wcls, file), config.vocab_size = -config.vocab_size;
        else wcls = token_embedding_table;

        #ifdef DEBUG
        std::cout << "Weights:"
            << "\n\ttoken_embedding_table=[" << token_embedding_table.size() << ", " << token_embedding_table[0].size() << "]"
            << "\n\trms_att_weight=[" << rms_att_weight.size() << ", " << rms_att_weight[0].size() << "]"
            << "\n\twq=[" << wq.size() << ", " << wq[0].size() << ", " << wq[0][0].size() << "]"
            << "\n\twk=[" << wk.size() << ", " << wk[0].size() << ", " << wk[0][0].size() << "]"
            << "\n\twv=[" << wv.size() << ", " << wv[0].size() << ", " << wv[0][0].size() << "]"
            << "\n\two=[" << wo.size() << ", " << wo[0].size() << ", " << wo[0][0].size() << "]"
            << "\n\trms_ffn_weight=[" << rms_ffn_weight.size() << ", " << rms_ffn_weight[0].size() << "]"
            << "\n\tw1=[" << w1.size() << ", " << w1[0].size() << ", " << w1[0][0].size() << "]"
            << "\n\tw2=[" << w2.size() << ", " << w2[0].size() << ", " << w2[0][0].size() << "]"
            << "\n\tw3=[" << w3.size() << ", " << w3[0].size() << ", " << w3[0][0].size() << "]"
            << "\n\trms_final_weight=[" << rms_final_weight.size() << "]"
            << "\n\tfreq_cis_real=[" << freq_cis_real.size() << ", " << freq_cis_real[0].size() << "]"
            << "\n\tfreq_cis_imag=[" << freq_cis_imag.size() << ", " << freq_cis_imag[0].size() << "]"
            << "\n\twcls=[" << wcls.size() << ", " << wcls[0].size() << "]" << std::endl;
        #endif
    }
};

struct Tokenizer {
    std::vector<std::string> vocab;
    std::vector<float> vocab_scores;
    std::vector<std::pair<int, std::string>> sorted_vocab;
    int max_token_length;
    void init(Config &config, std::fstream &file) {
        vocab.resize(config.vocab_size);
        vocab_scores.resize(config.vocab_size);
        sorted_vocab.resize(config.vocab_size);
        file.read(reinterpret_cast<char*>(&max_token_length), sizeof(int));
        // std::cout << "max_token_length=" << max_token_length << std::endl;
        for (int i = 0; i < config.vocab_size; i++) {
            int len;
            file.read(reinterpret_cast<char*>(&vocab_scores[i]), sizeof(float));
            file.read(reinterpret_cast<char*>(&len), sizeof(int));
            vocab[i] = "";
            for (int j = 0; j < len; ++j) {
                char c;
                file.read((char*)&c, sizeof(char));
                vocab[i].push_back(c);
            }
            sorted_vocab[i] = {i, vocab[i]};
            // std::cout << "vocab_scores[" << i << "]=" << vocab_scores[i] << " len=" << len << " vocab[" << i << "]=" << vocab[i] << std::endl;
        }
        std::sort(sorted_vocab.begin(), sorted_vocab.end(), [&](const auto &a, const auto &b) {
            return a.second < b.second;
        });

        #ifdef DEBUG
        std::cout << "Tokenizer:"
            << "\n\tmax_token_length=" << max_token_length
            << "\n\tvocab_scores=[" << vocab_scores.size() << "]"
            << "\n\tvocab=[" << vocab.size() << "]"
            << "\n\tsorted_vocab=[" << sorted_vocab.size() << "]" << std::endl;
        #endif
    }
};

int main(int argc, char** argv) {

    // 读取checkpoint(stories15M.bin)：Config token_embedding_table rms_att_weight wq wk wv wo rms_ffn_weight w1 w2 w3 rms_final_weight freq_cis_real freq_cis_imag wcls
    // Config: dim hidden_dim n_layers n_heads n_kv_heads vocab_size seq_len
    // Config: dim=288 hidden_dim=768 n_layers=6 n_heads=6 n_kv_heads=6 vocab_size=32000 seq_len=256
    // wcls根据shared_weights是否为true来决定是否和token_embedding_table共享内存，shared_weights根据vocab_size是否为正数来决定是否为true
    std::string checkpoint_path = "d:/project/llama2_cpp/model/stories15M.bin";
    std::string tokenizer_path = "d:/project/llama2_cpp/model/tokenizer.bin";

    Config config;
    Weights weights;
    {
        std::fstream file(checkpoint_path, std::ios::in | std::ios::out | std::ios::binary);
        if(!file) {
            std::cout << "Checkpoint File not found!" << std::endl;
            return 0;
        }
        config.init(file);
        weights.init(config, file);
        file.close();
    }

    // 读取分词器(tokenizer.bin)：max_token_length vocab_scores[i] len vocab[i]
    Tokenizer tokenizer;
    {
        std::fstream file(tokenizer_path, std::ios::in | std::ios::out | std::ios::binary);
        if(!file) {
            std::cout << "Tokenizer File not found!" << std::endl;
            return 0;
        }
        tokenizer.init(config, file);
        file.close();
    }

    return 0;
}