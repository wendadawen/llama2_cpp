#include <bits/stdc++.h>
#define DEBUG
#define EPS 1e-5

int GS = 0; // group size global for quantization of the weights

typedef std::vector<float> tensor1d;
typedef std::vector<tensor1d> tensor2d;
typedef std::vector<tensor2d> tensor3d;
struct qtensor1d {
    std::vector<int8_t> q;  // quantized values
    std::vector<float> s;  // scaling factors[q.len / GS]
    qtensor1d() = default;
    qtensor1d(int n) : q(n), s(n / GS) {}
    size_t size() const { return q.size(); }
};
typedef std::vector<qtensor1d> qtensor2d;
typedef std::vector<qtensor2d> qtensor3d;

void dequantize(qtensor1d &qx, tensor1d &x) {
    for (int i = 0; i < x.size(); i++) {
        x[i] = qx.q[i] * qx.s[i / GS];
    }
}
void dequantize(qtensor2d &qx, tensor2d &x) {
    for (int i = 0; i < x.size(); i++) {
        dequantize(qx[i], x[i]);
    }
}
void dequantize(qtensor3d &qx, tensor3d &x) {
    for (int i = 0; i < x.size(); i++) {
        dequantize(qx[i], x[i]);
    }
}

void quantize(qtensor1d &qx, tensor1d &x) {
    int num_groups = x.size() / GS;
    float Q_MAX = 127.0f;
    for (int group = 0; group < num_groups; group++) {
        float wmax = 0.0;
        for (int i = 0; i < GS; i++) {
            float val = fabs(x[group * GS + i]);
            if (val > wmax) {
                wmax = val;
            }
        }
        float scale = wmax / Q_MAX;
        qx.s[group] = scale;
        for (int i = 0; i < GS; i++) {
            float quant_value = x[group * GS + i] / scale; // scale
            int8_t quantized = (int8_t) round(quant_value); // round and clamp
            qx.q[group * GS + i] = quantized;
        }
    }
}
void quantize(qtensor2d &qx, tensor2d &x) {
    for (int i = 0; i < x.size(); i++) {
        quantize(qx[i], x[i]);
    }
}
void quantize(qtensor3d &qx, tensor3d &x) {
    for (int i = 0; i < x.size(); i++) {
        quantize(qx[i], x[i]);
    }
}

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
        // file.read(reinterpret_cast<char*>(this), sizeof(Config));  // struct直接对齐，sizeof(config)=44，实际我们希望是41
        file.read(reinterpret_cast<char*>(&magic_number), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&version), sizeof(int));
        file.read(reinterpret_cast<char*>(&dim), sizeof(int));
        file.read(reinterpret_cast<char*>(&hidden_dim), sizeof(int));
        file.read(reinterpret_cast<char*>(&n_layers), sizeof(int));
        file.read(reinterpret_cast<char*>(&n_heads), sizeof(int));
        file.read(reinterpret_cast<char*>(&n_kv_heads), sizeof(int));
        file.read(reinterpret_cast<char*>(&vocab_size), sizeof(int));
        file.read(reinterpret_cast<char*>(&seq_len), sizeof(int));
        file.read(reinterpret_cast<char*>(&shared_classifier), sizeof(uint8_t));
        file.read(reinterpret_cast<char*>(&group_size), sizeof(int));
        file.seekg(215, std::ios::cur);  // skip padding, header_size=256, 256 - 41 = 215

        GS = group_size;
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
            << "\n\tshared_classifier=" << (int)shared_classifier
            << "\n\tgroup_size=" << group_size << std::endl;
        #endif
    }
};

struct Weights {
    tensor2d token_embedding_table;  // [vocab_size, dim]
    tensor2d rms_att_weight;  // [layer, dim]
    tensor2d rms_ffn_weight;  // [layer, dim]
    tensor1d rms_final_weight;  // [dim]
    qtensor2d q_token_embedding_table;  // [vocab_size, dim]
    qtensor3d wq; // [layer, dim, dim]
    qtensor3d wk; // [layer, kv_dim, dim], where kv_dim = dim*n_kv_heads/n_heads
    qtensor3d wv; // [layer, kv_dim, dim]
    qtensor3d wo; // [layer, dim, dim]
    qtensor3d w1; // [layer, hidden_dim, dim]
    qtensor3d w2; // [layer, dim, hidden_dim]
    qtensor3d w3; // [layer, hidden_dim, dim]
    qtensor2d wcls;  // [vocab_size, dim]
    tensor2d freq_cis_real;  // [seq_len, head_size/2], where head_size=dim/n_heads
    tensor2d freq_cis_imag;  // [seq_len, head_size/2]
    void read_weight(tensor1d &tensor, std::fstream &file) {
        file.read(reinterpret_cast<char*>(tensor.data()), tensor.size() * sizeof(float));
    }
    void read_weight(tensor2d &tensor, std::fstream &file) {
        for (auto &t : tensor) read_weight(t, file);
    }
    void read_weight(tensor3d &tensor, std::fstream &file) {
        for (auto &t : tensor) read_weight(t, file);
    }
    void read_weight(qtensor2d &tensor, std::fstream &file) {
        for(auto &t : tensor) file.read(reinterpret_cast<char*>(t.q.data()), t.q.size() * sizeof(int8_t));
        for(auto &t : tensor) file.read(reinterpret_cast<char*>(t.s.data()), t.s.size() * sizeof(float));
    }
    void read_weight(qtensor3d &tensor, std::fstream &file) {
        for (auto &t : tensor) read_weight(t, file);
    }
    void init(Config &config, std::fstream &file) {
        token_embedding_table.resize(config.vocab_size, tensor1d(config.dim));
        rms_att_weight.resize(config.n_layers, tensor1d(config.dim));
        rms_ffn_weight.resize(config.n_layers, tensor1d(config.dim));
        rms_final_weight.resize(config.dim);
        q_token_embedding_table.resize(config.vocab_size, qtensor1d(config.dim));
        wq.resize(config.n_layers, qtensor2d(config.dim, qtensor1d(config.dim)));
        wq.resize(config.n_layers, qtensor2d(config.dim, qtensor1d(config.dim)));
        wk.resize(config.n_layers, qtensor2d(config.dim * config.n_kv_heads / config.n_heads, qtensor1d(config.dim)));
        wv.resize(config.n_layers, qtensor2d(config.dim * config.n_kv_heads / config.n_heads, qtensor1d(config.dim)));
        wo.resize(config.n_layers, qtensor2d(config.dim, qtensor1d(config.dim)));
        w1.resize(config.n_layers, qtensor2d(config.hidden_dim, qtensor1d(config.dim)));
        w2.resize(config.n_layers, qtensor2d(config.dim, qtensor1d(config.hidden_dim)));
        w3.resize(config.n_layers, qtensor2d(config.hidden_dim, qtensor1d(config.dim)));
        wcls.resize(config.vocab_size, qtensor1d(config.dim));
        freq_cis_real.resize(config.seq_len, tensor1d(config.dim / config.n_heads / 2));
        freq_cis_imag.resize(config.seq_len, tensor1d(config.dim / config.n_heads / 2));

        read_weight(rms_att_weight, file);
        read_weight(rms_ffn_weight, file);
        read_weight(rms_final_weight, file);
        read_weight(q_token_embedding_table, file);
        read_weight(wq, file);
        read_weight(wk, file);
        read_weight(wv, file);
        read_weight(wo, file);
        read_weight(w1, file);
        read_weight(w2, file);
        read_weight(w3, file);
        if((int)config.shared_classifier != 1) read_weight(wcls, file);
        else wcls = q_token_embedding_table;

        dequantize(q_token_embedding_table, token_embedding_table);
        int head_size = config.dim / config.n_heads;
        int half_head_size = head_size / 2;
        std::vector<float> base_freq(half_head_size);
        for (int j = 0; j < half_head_size; j++) {
            base_freq[j] = 1.0f / powf(10000.0f, 2.0f * j / (float)head_size);
        }
        for (int i = 0; i < config.seq_len; i++) {
            for (int j = 0; j < half_head_size; j++) {
                float angle = i * base_freq[j];
                freq_cis_real[i][j] = cosf(angle);
                freq_cis_imag[i][j] = sinf(angle);
            }
        }

        #ifdef DEBUG
        std::cerr << "Weights:"
            << "\n\ttoken_embedding_table=[" << token_embedding_table.size() << ", " << token_embedding_table[0].size() << "]"
            << "\n\trms_att_weight=[" << rms_att_weight.size() << ", " << rms_att_weight[0].size() << "]"
            << "\n\trms_ffn_weight=[" << rms_ffn_weight.size() << ", " << rms_ffn_weight[0].size() << "]"
            << "\n\trms_final_weight=[" << rms_final_weight.size() << "]"
            << "\n\twq=[" << wq.size() << ", " << wq[0].size() << ", " << wq[0][0].size() << "]"
            << "\n\twk=[" << wk.size() << ", " << wk[0].size() << ", " << wk[0][0].size() << "]"
            << "\n\twv=[" << wv.size() << ", " << wv[0].size() << ", " << wv[0][0].size() << "]"
            << "\n\two=[" << wo.size() << ", " << wo[0].size() << ", " << wo[0][0].size() << "]"
            << "\n\tw1=[" << w1.size() << ", " << w1[0].size() << ", " << w1[0][0].size() << "]"
            << "\n\tw2=[" << w2.size() << ", " << w2[0].size() << ", " << w2[0][0].size() << "]"
            << "\n\tw3=[" << w3.size() << ", " << w3[0].size() << ", " << w3[0][0].size() << "]"
            << "\n\twcls=[" << wcls.size() << ", " << wcls[0].size() << "]"
            << "\n\tfreq_cis_real=[" << freq_cis_real.size() << ", " << freq_cis_real[0].size() << "]"
            << "\n\tfreq_cis_imag=[" << freq_cis_imag.size() << ", " << freq_cis_imag[0].size() << "]" 
            << std::endl;
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
        }
        std::sort(sorted_vocab.begin(), sorted_vocab.end(), [&](const auto &a, const auto &b) {
            return a.second < b.second;
        });

        #ifdef DEBUG
        std::cerr << "Tokenizer:"
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
    tensor1d k;  // [kv_dim]
    tensor1d v;  // [kv_dim]
    tensor3d key_cache;  // [layer, seq_len, kv_dim]
    tensor3d value_cache;  // [layer, seq_len, kv_dim]
    tensor1d attention;  // [seq_len]
    tensor1d logits;  // [vocab_size]
    qtensor1d xq;  // [dim]
    qtensor1d hq;  // [hidden_dim]

    void init(Config &config) {
        x.resize(config.dim);
        xb.resize(config.dim);
        xb2.resize(config.dim);
        hb.resize(config.hidden_dim);
        hb2.resize(config.hidden_dim);
        q.resize(config.dim);
        k.resize(config.dim * config.n_kv_heads / config.n_heads);
        v.resize(config.dim * config.n_kv_heads / config.n_heads);
        key_cache.resize(config.n_layers, tensor2d(config.seq_len, tensor1d(config.dim * config.n_kv_heads / config.n_heads)));
        value_cache.resize(config.n_layers, tensor2d(config.seq_len, tensor1d(config.dim * config.n_kv_heads / config.n_heads)));
        attention.resize(config.seq_len);
        logits.resize(config.vocab_size);
        xq = qtensor1d(config.dim);
        hq = qtensor1d(config.hidden_dim);
        #ifdef DEBUG
        std::cerr << "RunState:"
            << "\n\tx=[" << x.size() << "]"
            << "\n\txb=[" << xb.size() << "]"
            << "\n\txb2=[" << xb2.size() << "]"
            << "\n\thb=[" << hb.size() << "]"
            << "\n\thb2=[" << hb2.size() << "]"
            << "\n\tq=[" << q.size() << "]"
            << "\n\tk=[" << k.size() << "]"
            << "\n\tv=[" << v.size() << "]"
            << "\n\tkey_cache=[" << key_cache.size() << ", " << key_cache[0].size() << ", " << key_cache[0][0].size() << "]"
            << "\n\tvalue_cache=[" << value_cache.size() << ", " << value_cache[0].size() << ", " << value_cache[0][0].size() << "]"
            << "\n\tattention=[" << attention.size() << "]"
            << "\n\tlogits=[" << logits.size() << "]" 
            << "\n\txq=[" << xq.size() << ", " << xq.s.size() << "]"
            << "\n\thq=[" << hq.size() << ", " << hq.s.size() << "]"
            << std::endl;
        #endif
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
    unsigned long long rng_seed;
    std::string prompt;
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
        if(cumulative_prob > topp) {
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
        float coin = random_f32(transformer.rng_seed);
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
    rms = 1.0f / sqrt(rms);
    for(int i = 0; i < n; i ++) {
        output[i] = weight[i] * (rms * input[i]);
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
void matmul(tensor1d &output, qtensor1d &input, qtensor2d &weight) {
    int n = input.size();
    for (int i = 0; i < output.size(); i++) {
        float val = 0.0f;
        int32_t ival = 0;
        for (int j = 0; j <= n - GS; j += GS) {
            for (int k = 0; k < GS; k++) {
                ival += ((int32_t) input.q[j + k]) * ((int32_t) weight[i].q[j + k]);
            }
            val += ((float) ival) * weight[i].s[j / GS] * input.s[j / GS];
            ival = 0;
        }
        output[i] = val;
    }
}

void rope(RunState &state, int pos, Config &config, tensor2d &freq_cis_real, tensor2d &freq_cis_imag) {
    int head_size = config.dim / config.n_heads;
    for (int head = 0; head < config.n_heads; ++head) {
        int start = head * head_size;
        for (int i = 0; i < head_size; i += 2) {
            float fcr = freq_cis_real[pos][i / 2];
            float fci = freq_cis_imag[pos][i / 2];
            float q0 = state.q[start + i];
            float q1 = state.q[start + i + 1];
            state.q[start + i]     = q0 * fcr - q1 * fci;
            state.q[start + i + 1] = q0 * fci + q1 * fcr;
            if(start + i < state.k.size()) {
                float k0 = state.k[start + i];
                float k1 = state.k[start + i + 1];
                state.k[start + i]     = k0 * fcr - k1 * fci;
                state.k[start + i + 1] = k0 * fci + k1 * fcr;
            }
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
    int kv_mul = config.n_heads / config.n_kv_heads;

    state.x = weights.token_embedding_table[token];
    for(int layer = 0; layer < config.n_layers; ++ layer) {
        rmsnorm(state.xb, state.x, weights.rms_att_weight[layer]);
        quantize(state.xq, state.xb);
        matmul(state.q, state.xq, weights.wq[layer]);
        matmul(state.k, state.xq, weights.wk[layer]);
        matmul(state.v, state.xq, weights.wv[layer]);
        rope(state, pos, config, weights.freq_cis_real, weights.freq_cis_imag);
        state.key_cache[layer][pos] = state.k;
        state.value_cache[layer][pos] = state.v;
        memset(state.xb.data(), 0, state.xb.size() * sizeof(float));
        for(int head = 0; head < config.n_heads; ++ head) {
            int offest1 = head * head_size;
            int offest2 = head / kv_mul * head_size;
            for(int step = 0; step <= pos; ++ step) {
                float score = 0;
                for(int i = 0; i < head_size; ++ i) {
                    score += state.q[offest1 + i] * state.key_cache[layer][step][offest2 + i];
                }
                score /= sqrt(head_size * 1.0);
                state.attention[step] = score;
            }
            softmax(state.attention, state.attention, pos+1);
            for(int step = 0; step <= pos; ++ step) {
                float a = state.attention[step];
                for(int i = 0; i < head_size; ++ i) {
                    state.xb[offest1 + i] += a * state.value_cache[layer][step][offest2 + i];
                }
            }
        }
        quantize(state.xq, state.xb);
        matmul(state.xb2, state.xq, weights.wo[layer]);
        accumulate(state.x, state.xb2);
        rmsnorm(state.xb, state.x, weights.rms_ffn_weight[layer]);
        quantize(state.xq, state.xb);
        matmul(state.hb, state.xq, weights.w1[layer]);
        matmul(state.hb2, state.xq, weights.w3[layer]);
        for (int i = 0; i < config.hidden_dim; ++i)
            state.hb[i] = state.hb[i] * (1.0 / (1.0 + std::exp(-state.hb[i]))) * state.hb2[i];
        quantize(state.hq, state.hb);
        matmul(state.xb, state.hq, weights.w2[layer]);
        accumulate(state.x, state.xb);
    }
    rmsnorm(state.x, state.x, weights.rms_final_weight);
    quantize(state.xq, state.x);
    matmul(state.logits, state.xq, weights.wcls);
}

int str_lookup(const std::string &str, const std::vector<std::pair<int, std::string>> &sorted_vocab) {
    std::pair<int, std::string> key = {-1, str};
    auto it = std::lower_bound(sorted_vocab.begin(), sorted_vocab.end(), key, [&](const auto &a, const auto &b) {
        return a.second < b.second;
    });
    if (it != sorted_vocab.end() && it->second == str) {
        return it->first;
    }
    return -1;
}

std::vector<int> encode(Tokenizer &t, const std::string &text, bool bos, bool eos) {
    std::vector<int> tokens;
    if (bos) tokens.push_back(1); // BOS token
    if (text.empty()) return tokens;
    else {
        int dummy_prefix = str_lookup(" ", t.sorted_vocab);
        tokens.push_back(dummy_prefix);
    }
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx
    std::string str_buffer;
    for (size_t i = 0; i < text.size(); ++i) {
        char c = text[i];
        // 0xC0 is 11000000
        // 0x80 is 10000000
        if ((c & 0xC0) != 0x80) str_buffer.clear();
        str_buffer.push_back(c);
        // 如果下一个字符是延续字节，继续读取
        if (i + 1 < text.size() && (text[i + 1] & 0xC0) == 0x80 && str_buffer.size() < 4) {
            continue;
        }
        int id = str_lookup(str_buffer, t.sorted_vocab);
        if (id != -1) {
            tokens.push_back(id);
        } else {
            for (char ch : str_buffer) tokens.push_back(static_cast<unsigned char>(ch) + 3);
        }
        str_buffer.clear();
    }
    // 合并最佳连续对
    while (true) {
        float best_score = -1e10f;
        int best_id = -1;
        size_t best_idx = -1;
        for (size_t i = 0; i + 1 < tokens.size(); ++i) {
            std::string merged_str = t.vocab[tokens[i]] + t.vocab[tokens[i + 1]];
            int id = str_lookup(merged_str, t.sorted_vocab);
            if (id != -1 && t.vocab_scores[id] > best_score) {
                best_score = t.vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }
        if (best_idx == static_cast<size_t>(-1)) break;
        tokens[best_idx] = best_id;
        tokens.erase(tokens.begin() + best_idx + 1);
    }
    if (eos) tokens.push_back(2);
    return tokens;
}

std::string decode(Tokenizer &t, int prev_token, int token) {
    std::string piece = t.vocab[token];
    if (prev_token == 1 && piece[0] == ' ') piece = piece.substr(1);
    unsigned char byte_val;
    if (sscanf(piece.c_str(), "<0x%02hhX>", &byte_val) == 1) {
        piece = (char)byte_val;
    }
    return piece;
}

void safe_cout(const std::string& piece) {
    if (piece.empty()) { return; }
    if (piece.size() == 1) {
        unsigned char byte_val = piece[0];
        if (!(std::isprint(byte_val) || std::isspace(byte_val))) {
            return;
        }
    }
    std::cout << piece;
}

void generate(Transformer &transformer) {
    std::vector<int> prompt_tokens = encode(transformer.tokenizer, transformer.prompt, 1, 0);
    if (prompt_tokens.size() < 1) {
        std::cerr << "something is wrong, expected at least 1 prompt token" << std::endl;
        exit(EXIT_FAILURE);
    }
    int next;
    int token = prompt_tokens[0];  // BOS = 1, EOS = 2
    for(int pos = 0; pos < transformer.steps; pos ++) {
        forward(transformer, token, pos);
        if (pos < prompt_tokens.size() - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(transformer);
        }
        if(next == 1) break;  // BOS
        safe_cout(decode(transformer.tokenizer, token, next));
        token = next;
    }
    std::cout << std::endl;
}

void error_usage() {
    fprintf(stderr, "Usage:   main <checkpoint> [options]\n");
    fprintf(stderr, "Example: main model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char** argv) {

    // 读取checkpoint： Config rms_att_weight(float) rms_ffn_weight(float) rms_final_weight(float) q_token_embedding_table(int8, float) wq(layers int8, float) wk(layers int8, float) wv(layers int8, float) wo(layers int8, float) w1(layers int8, float) w2(layers int8, float) w3(layers int8, float) wcls(int8, float, optional)
    std::string checkpoint_path = "d:/project/llama2_cpp/model/stories15Mq.bin";
    std::string tokenizer_path = "d:/project/llama2_cpp/model/tokenizer.bin";
    
    float temperature = 1.0; // t = 0, greedy; t = 1.0, sampling; t > 1.0, sampling with more randomness
    int steps = 256;  // 生成的token数量
    float topp = 0.9;  // top-p sampling
    unsigned long long rng_seed = (unsigned int)time(NULL);
    std::string prompt = "";
    std::string mode = "generate";
    std::string system_prompt = "";

    if (argc >= 2) { checkpoint_path = "d:/project/llama2_cpp/model/" + std::string(argv[1]); } 
    else if (argc == 1){} 
    else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        if (i + 1 >= argc) { error_usage(); }
        if (argv[i][0] != '-') { error_usage(); }
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = "d:/project/llama2_cpp/model/"+std::string(argv[i + 1]); }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else { error_usage(); }
    }

    Config config;
    Weights weights;
    {
        std::fstream file(checkpoint_path, std::ios::in | std::ios::out | std::ios::binary);
        if(!file) {
            std::cerr << "Checkpoint File not found!" << std::endl;
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
            std::cerr << "Tokenizer File not found!" << std::endl;
            return 0;
        }
        tokenizer.init(config, file);
        file.close();
    }

    RunState state; 
    state.init(config);

    Transformer transformer = {config, weights, tokenizer, state, temperature, steps, topp, rng_seed, prompt};
    #ifdef DEBUG
    std::cerr << "Transformer:"
        << "\n\ttemperature=" << temperature
        << "\n\tsteps=" << steps
        << "\n\ttopp=" << topp
        << "\n\trng_seed=" << rng_seed
        << "\n\tprompt=" << prompt << std::endl;
    #endif
    generate(transformer);

    return 0;
}