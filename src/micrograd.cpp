#include <bits/stdc++.h>
using namespace std;

class Value {
   private:
    struct Node {
        double data;
        double grad;
        vector<shared_ptr<Node>> prev;
        string op;
        string label;
        function<void()> backward;

        Node(
            double val,
            vector<shared_ptr<Node>> children = {},
            string operation = "",
            string node_label = "")
            : data(val),
              grad(0.0),
              prev(move(children)),
              op(move(operation)),
              label(move(node_label)),
              backward([] {}) {}
    };

    shared_ptr<Node> node;

    explicit Value(shared_ptr<Node> node) : node(move(node)) {}

    static Value make_node(double data, vector<shared_ptr<Node>> children, const string& op) {
        return Value(make_shared<Node>(data, move(children), op));
    }

    static void build_topo(
        const shared_ptr<Node>& current,
        unordered_set<Node*>& visited,
        vector<shared_ptr<Node>>& topo) {
        if (visited.count(current.get())) {
            return;
        }

        visited.insert(current.get());
        for (const auto& child : current->prev) {
            build_topo(child, visited, topo);
        }
        topo.push_back(current);
    }

   public:
    Value(double val) : node(make_shared<Node>(val)) {}
    Value(double val, const string& label) : node(make_shared<Node>(val, vector<shared_ptr<Node>>{}, "", label)) {}

    double data() const {
        return node->data;
    }

    double grad() const {
        return node->grad;
    }

    void set_label(const string& label) {
        node->label = label;
    }

    void set_data(double data) {
        node->data = data;
    }

    void zero_grad() {
        node->grad = 0.0;
    }

    Value operator+(const Value& other) const {
        auto lhs = node;
        auto rhs = other.node;
        auto out = make_node(lhs->data + rhs->data, {lhs, rhs}, "+");
        weak_ptr<Node> out_node = out.node;

        out.node->backward = [lhs, rhs, out_node]() {
            if (auto out = out_node.lock()) {
                lhs->grad += out->grad;
                rhs->grad += out->grad;
            }
        };

        return out;
    }

    Value operator-() const {
        return *this * -1.0;
    }

    Value operator-(const Value& other) const {
        return *this + (-other);
    }

    Value operator*(const Value& other) const {
        auto lhs = node;
        auto rhs = other.node;
        auto out = make_node(lhs->data * rhs->data, {lhs, rhs}, "*");
        weak_ptr<Node> out_node = out.node;

        out.node->backward = [lhs, rhs, out_node]() {
            if (auto out = out_node.lock()) {
                lhs->grad += rhs->data * out->grad;
                rhs->grad += lhs->data * out->grad;
            }
        };

        return out;
    }

    Value pow(double exponent) const {
        auto base = node;
        auto out = make_node(std::pow(base->data, exponent), {base}, "**" + to_string(exponent));
        weak_ptr<Node> out_node = out.node;

        out.node->backward = [base, exponent, out_node]() {
            if (auto out = out_node.lock()) {
                base->grad += exponent * std::pow(base->data, exponent - 1.0) * out->grad;
            }
        };

        return out;
    }

    Value exp() const {
        auto input = node;
        auto out = make_node(std::exp(input->data), {input}, "exp");
        weak_ptr<Node> out_node = out.node;

        out.node->backward = [input, out_node]() {
            if (auto out = out_node.lock()) {
                input->grad += out->data * out->grad;
            }
        };

        return out;
    }

    Value tanh() const {
        auto input = node;
        double t = std::tanh(input->data);
        auto out = make_node(t, {input}, "tanh");
        weak_ptr<Node> out_node = out.node;

        out.node->backward = [input, t, out_node]() {
            if (auto out = out_node.lock()) {
                input->grad += (1.0 - t * t) * out->grad;
            }
        };

        return out;
    }

    Value operator/(const Value& other) const {
        return *this * other.pow(-1.0);
    }

    Value operator+(double num) const {
        return *this + Value(num);
    }

    Value operator-(double num) const {
        return *this - Value(num);
    }

    Value operator*(double num) const {
        return *this * Value(num);
    }

    Value operator/(double num) const {
        return *this / Value(num);
    }

    void backward() {
        vector<shared_ptr<Node>> topo;
        unordered_set<Node*> visited;

        build_topo(node, visited, topo);
        node->grad = 1.0;

        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            (*it)->backward();
        }
    }

    friend Value operator+(double num, const Value& v) {
        return Value(num) + v;
    }

    friend Value operator-(double num, const Value& v) {
        return Value(num) - v;
    }

    friend Value operator*(double num, const Value& v) {
        return Value(num) * v;
    }

    friend Value operator/(double num, const Value& v) {
        return Value(num) / v;
    }

    friend ostream& operator<<(ostream& os, const Value& v) {
        os << v.node->data;
        return os;
    }
};

class Neuron {
   private:
    vector<Value> w;
    Value b;

    static double random_uniform(double low = -1.0, double high = 1.0) {
        static random_device rd;
        static mt19937 gen(rd());
        uniform_real_distribution<double> dist(low, high);
        return dist(gen);
    }

   public:
    explicit Neuron(int nin) : b(random_uniform()) {
        w.reserve(nin);
        for (int i = 0; i < nin; i++) {
            w.emplace_back(random_uniform());
        }
    }

    Value operator()(const vector<Value>& x) const {
        if (x.size() != w.size()) {
            throw invalid_argument("Neuron input size must match number of weights");
        }

        Value act = b;
        for (size_t i = 0; i < w.size(); i++) {
            act = act + w[i] * x[i];
        }
        return act.tanh();
    }

    vector<Value> parameters() const {
        vector<Value> params = w;
        params.push_back(b);
        return params;
    }
};

class Layer {
   private:
    vector<Neuron> neurons;

   public:
    Layer(int nin, int nout) {
        neurons.reserve(nout);
        for (int i = 0; i < nout; i++) {
            neurons.emplace_back(nin);
        }
    }

    vector<Value> operator()(const vector<Value>& x) const {
        vector<Value> outs;
        outs.reserve(neurons.size());
        for (const auto& neuron : neurons) {
            outs.push_back(neuron(x));
        }
        return outs;
    }

    vector<Value> parameters() const {
        vector<Value> params;
        for (const auto& neuron : neurons) {
            vector<Value> neuron_params = neuron.parameters();
            params.insert(params.end(), neuron_params.begin(), neuron_params.end());
        }
        return params;
    }
};

class MLP {
   private:
    vector<Layer> layers;

   public:
    MLP(int nin, const vector<int>& nouts) {
        vector<int> sizes = {nin};
        sizes.insert(sizes.end(), nouts.begin(), nouts.end());

        layers.reserve(nouts.size());
        for (size_t i = 0; i < nouts.size(); i++) {
            layers.emplace_back(sizes[i], sizes[i + 1]);
        }
    }

    vector<Value> operator()(vector<Value> x) const {
        for (const auto& layer : layers) {
            x = layer(x);
        }
        return x;
    }

    vector<Value> operator()(const vector<double>& x) const {
        vector<Value> values;
        values.reserve(x.size());
        for (double val : x) {
            values.emplace_back(val);
        }
        return (*this)(values);
    }

    vector<Value> parameters() const {
        vector<Value> params;
        for (const auto& layer : layers) {
            vector<Value> layer_params = layer.parameters();
            params.insert(params.end(), layer_params.begin(), layer_params.end());
        }
        return params;
    }
};

int main() {
    MLP n(3, {4, 4, 1});

    vector<vector<double>> xs = {
        {2.0, 3.0, -1.0},
        {3.0, -1.0, 0.5},
        {0.5, 1.0, 1.0},
        {1.0, 1.0, -1.0},
    };
    vector<double> ys = {1.0, -1.0, -1.0, 1.0};

    for (int k = 0; k < 50; k++) {
        vector<Value> ypred;
        ypred.reserve(xs.size());
        for (const auto& x : xs) {
            ypred.push_back(n(x)[0]);
        }

        Value loss(0.0);
        for (size_t i = 0; i < ys.size(); i++) {
            loss = loss + (ypred[i] - ys[i]).pow(2.0);
        }

        vector<Value> params = n.parameters();
        for (auto& p : params) {
            p.zero_grad();
        }

        loss.backward();

        for (auto& p : params) {
            p.set_data(p.data() - 0.05 * p.grad());
        }

        cout << k << " " << loss.data() << endl;
    }

    cout << "ypred" << endl;
    for (const auto& x : xs) {
        cout << n(x)[0] << endl;
    }

    return 0;
}
