#include <bits/stdc++.h>
#define DEBUG
#define EPS 1e-5

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
    // 值得注意的是，这里的weight的存储是转置的（如wk应该是[dim, kv_dim]），详情参考pytorch的Linear层的权重存储
    tensor3d wq; // [layer, dim, dim]
    tensor3d wk; // [layer, kv_dim, dim], where kv_dim = dim*n_kv_heads/n_heads
    tensor3d wv; // [layer, kv_dim, dim]
    tensor3d wo; // [layer, dim, dim]
    tensor2d rms_ffn_weight;  // [layer, dim]
    tensor3d w1; // [layer, hidden_dim, dim]
    tensor3d w2; // [layer, dim, hidden_dim]
    tensor3d w3; // [layer, hidden_dim, dim]
    tensor1d rms_final_weight;  // [dim]
    tensor2d freq_cis_real;  // [seq_len, head_size/2], where head_size=dim/n_heads
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

struct RunState {
    tensor1d x;  // [dim]
    tensor1d xb;  // [dim]
    tensor1d xb2;  // [dim]
    tensor1d hb;  // [hidden_dim]
    tensor1d hb2;  // [hidden_dim]
    tensor1d q;  // [dim]
    tensor1d k;  // [dim]
    tensor1d v;  // [dim]
    tensor3d key_cache;  // [layer, seq_len, dim]
    tensor3d value_cache;  // [layer, seq_len, dim]
    tensor1d attention;  // [seq_len]
    tensor1d logits;  // [vocab_size]

    void init(Config &config) {
        x.resize(config.dim);
        xb.resize(config.dim);
        xb2.resize(config.dim);
        hb.resize(config.hidden_dim);
        hb2.resize(config.hidden_dim);
        q.resize(config.dim);
        k.resize(config.dim);
        v.resize(config.dim);
        key_cache.resize(config.n_layers, tensor2d(config.seq_len, tensor1d(config.dim)));
        value_cache.resize(config.n_layers, tensor2d(config.seq_len, tensor1d(config.dim)));
        attention.resize(config.seq_len);
        logits.resize(config.vocab_size);
    }
};

struct Transformer {
    Config config;
    Weights weights;
    Tokenizer tokenizer;
    RunState state;

    float temperature;
    int steps;
    float topp;
    unsigned long long rng_state;
};

void softmax(tensor1d &output, tensor1d &input, int n = -1) {
    if(n < 0) n = input.size();
    float max_val = *std::max_element(input.begin(), input.begin() + n);
    float sum = 0;
    for(int i = 0; i < n; i ++) {
        output[i] = exp(input[i] - max_val);
        sum += output[i];
    }
    for(int i = 0; i < n; i ++) {
        output[i] /= sum;
    }
}

unsigned int random_u32(unsigned long long &state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return (state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long &state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int argmax(tensor1d &logits) {
    return std::max_element(logits.begin(), logits.end()) - logits.begin();
}

int sample_mult(tensor1d &logits, float coin) {
    float cdf = 0.0f;
    for(int i = 0; i < logits.size(); i ++) {
        cdf += logits[i];
        if(coin < cdf) {
            return i;
        }
    }
    return logits.size() - 1;
}

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

int sample(Transformer &transformer) {
    int next;
    if(transformer.temperature < EPS) {
        next = argmax(transformer.state.logits);
    } else {
        for(int q = 0; q < transformer.config.vocab_size; q ++) {
            transformer.state.logits[q] /= transformer.temperature;
        }
        softmax(transformer.state.logits, transformer.state.logits);
        float coin = random_f32(transformer.rng_state);
        if(transformer.topp <= 0 || transformer.topp >= 1) {
            next = sample_mult(transformer.state.logits, coin);
        } else {
            next = sample_topp(transformer.state.logits, transformer.topp, coin);
        }
    }
    return next;
}

void rmsnorm(tensor1d &output, tensor1d &input, tensor1d &weight) {
    float rms = 0.0f;
    int n = input.size();
    for(int i = 0; i < n; i ++) {
        rms += input[i] * input[i];
    }
    rms = rms / n + EPS;
    rms = 1 / sqrt(rms);
    for(int i = 0; i < n; i ++) {
        output[i] = (input[i] * rms) * weight[i];
    }
}

void matmul(tensor1d &output, tensor1d &input, tensor2d &weight) {
    // output = input[dim] * weight[dim, dim/kv_dim]
    for(int i = 0; i < output.size(); i ++) {
        output[i] = 0;
        for(int j = 0; j < input.size(); j ++) {
            output[i] += input[j] * weight[i][j];
        }
    }
}

void rope(RunState &state, int pos, Config &config, tensor2d &freq_cis_real, tensor2d &freq_cis_imag) {
    int head_size = config.dim / config.n_heads;
    for (int head = 0; head < config.n_heads; ++head) {
        int start = head * head_size;
        for (int i = 0; i < head_size; i += 2) {
            float q0 = state.q[start + i];
            float q1 = state.q[start + i + 1];
            float k0 = state.k[start + i];
            float k1 = state.k[start + i + 1];
            float fcr = freq_cis_real[pos][i / 2];
            float fci = freq_cis_imag[pos][i / 2];
            state.q[start + i]     = q0 * fcr - q1 * fci;
            state.q[start + i + 1] = q0 * fci + q1 * fcr;
            state.k[start + i]     = k0 * fcr - k1 * fci;
            state.k[start + i + 1] = k0 * fci + k1 * fcr;
        }
    }
}

void accumulate(tensor1d &output, tensor1d &input) {
    for(int i = 0; i < output.size(); i ++) {
        output[i] += input[i];
    }
}

void forward(Transformer &transformer, int token, int pos) {
    Config &config = transformer.config;
    Weights &weights = transformer.weights;
    RunState &state = transformer.state;
    Tokenizer &tokenizer = transformer.tokenizer;
    int head_size = config.dim / config.n_heads;

    state.x = weights.token_embedding_table[token];
    for(int layer = 0; layer < config.n_layers; ++ layer) {
        rmsnorm(state.xb, state.x, weights.rms_att_weight[layer]);
        matmul(state.q, state.xb, weights.wq[layer]);
        matmul(state.k, state.xb, weights.wk[layer]);
        matmul(state.v, state.xb, weights.wv[layer]);
        rope(state, pos, config, weights.freq_cis_real, weights.freq_cis_imag);
        state.key_cache[layer][pos] = state.k;
        state.value_cache[layer][pos] = state.v;
        memset(state.xb.data(), 0, state.xb.size() * sizeof(float));
        for(int head = 0; head < config.n_heads; ++ head) {
            int offest = head * head_size;
            for(int step = 0; step <= pos; ++ step) {
                float score = 0;
                for(int i = 0; i < head_size; ++ i) {
                    score += state.q[offest + i] * state.key_cache[layer][step][offest + i];
                }
                score /= sqrt(head_size * 1.0);
                state.attention[step] = score;
            }
            softmax(state.attention, state.attention, pos+1);
            for(int step = 0; step <= pos; ++ step) {
                float a = state.attention[step];
                for(int i = 0; i < head_size; ++ i) {
                    state.xb[offest + i] += a * state.value_cache[layer][step][offest + i];
                }
            }
        }
        matmul(state.xb2, state.xb, weights.wo[layer]);
        accumulate(state.x, state.xb2);
        rmsnorm(state.xb, state.x, weights.rms_ffn_weight[layer]);
        matmul(state.hb, state.xb, weights.w1[layer]);
        matmul(state.hb2, state.xb, weights.w3[layer]);
        for (int i = 0; i < config.hidden_dim; ++i)
            state.hb[i] = state.hb[i] * (1.0 / (1.0 + std::exp(-state.hb[i]))) * state.hb2[i];
        matmul(state.xb, state.hb, weights.w2[layer]);
        accumulate(state.x, state.xb);
    }
    rmsnorm(state.x, state.x, weights.rms_final_weight);
    matmul(state.logits, state.x, weights.wcls);
}

void generate(Transformer &transformer) {
    int next;
    int token = 1;  // BOS = 1, EOS = 2
    for(int pos = 0; pos < transformer.steps; pos ++) {
        forward(transformer, token, pos);
        next = sample(transformer);
        if(next == 1) break;  // BOS
        std::cout << transformer.tokenizer.vocab[next];
        token = next;
    }
}

int main(int argc, char** argv) {

    // 读取checkpoint(stories15M.bin)：Config token_embedding_table rms_att_weight wq wk wv wo rms_ffn_weight w1 w2 w3 rms_final_weight freq_cis_real freq_cis_imag wcls
    // Config: dim hidden_dim n_layers n_heads n_kv_heads vocab_size seq_len
    // Config: dim=288 hidden_dim=768 n_layers=6 n_heads=6 n_kv_heads=6 vocab_size=32000 seq_len=256
    // wcls根据shared_weights是否为true来决定是否和token_embedding_table共享内存，shared_weights根据vocab_size是否为正数来决定是否为true
    std::string checkpoint_path = "d:/project/llama2_cpp/model/stories15M.bin";
    std::string tokenizer_path = "d:/project/llama2_cpp/model/tokenizer.bin";
    float temperature = 1.0; // t = 0, greedy; t = 1.0, sampling; t > 1.0, sampling with more randomness
    int steps = 256;  // 生成的token数量
    float topp = 0.9;  // top-p sampling
    unsigned long long rng_state = (unsigned int)time(NULL);

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

    RunState state; 
    state.init(config);

    Transformer transformer = {config, weights, tokenizer, state, temperature, steps, topp, rng_state};
    generate(transformer);

    return 0;
}