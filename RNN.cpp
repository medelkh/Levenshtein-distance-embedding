#include "matrix.cpp"

struct network{
    int len;
    int dim;
    int layer_dim;
    double learning_rate;
    double momentum;
    Matrix W,H,F;
    Matrix input[2][3], layer[2][3], ac_layer[2][3];
    Matrix layer_bias;
    Matrix embedding[2][3], ac_embedding[2][3]; //vecteur image
    Matrix grad_errW, grad_errH, grad_errF, grad_bias; //gradients
    string input_str[2];
    int edit_dist[3];
    double embed_dist[3];
    string data_path;
    double scale;
    int batch_size=2;    
    network(int string_len, int final_dim, int layerdim=-1, double speed, double mom){
        if(layerdim==-1) layer_dim = string_len;
        else layer_dim = layerdim;
        len = string_len;
        dim = final_dim;
        W = Matrix::Random(layer_dim, len);
        H = Matrix::Random(layer_dim,layer_dim);
        F = Matrix::Random(dim, layer_dim);
        grad_errW = Matrix::Zero(layer_dim, len);
        grad_errH = Matrix::Zero(layer_dim,layer_dim);
        grad_errF = Matrix::Zero(dim, layer_dim);
        learning_rate = speed;
        layer_bias = Matrix::Random(layer_dim,1);
        grad_bias = Matrix::Zero(layer_dim,1);
        scale = ((double)(len)*3.)/(4.*sqrt(dim));
        momentum = mom;
        data_path = "data"+to_string(len)+"_"+to_string(dim)+".txt";
    }

    void clear_input(){
        for(int i=0;i<2;i++){
            for(int j=0;j<3;j++){
                input[i][j] = Matrix::Zero(len,1);
            }
        }   
    }

    double levenshtein(string a, string b){
        int n = a.length(), m = b.length();
        int dp[n+1][m+1];
        for(int i=0;i<=n;i++) dp[i][0]=i;
        for(int i=1;i<=m;i++) dp[0][i]=i;
        for(int i=1;i<=n;i++){
            for(int j=1;j<=m;j++){
                dp[i][j] = min({dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+int(a[i-1]!=b[j-1])});
            }
        }
        return (double)(dp[n][m]);
    }

    double deriv(double x){
        return (1-tanh(x)*tanh(x))/(double)(2.5*len);
    }

    Matrix tanhM(Matrix M){
        for(int i=0;i<M.rows();i++)
            for(int j=0;j<M.cols();j++)
                M(i,j) = tanh(M(i,j)/(2.5*len));
        return M;
    }

    Matrix derivM(Matrix M){
        for(int i=0;i<M.rows();i++)
            for(int j=0;j<M.cols();j++)
                M(i,j) = deriv(M(i,j));
        return M;
    }

    void add_input(string sq, int ind=0){
        int n = sq.length();
        for(int i=0;i<n;i++){
            if(sq[i]=='A') input[ind][0](i,0)=1;
            else if(sq[i]=='G') input[ind][1](i,0)=1;
            else if(sq[i]=='C') input[ind][2](i,0)=1;
        }
        input_str[ind] = sq;
    }

    void execute(int ind){
        layer[ind][0] = (W * input[ind][0]) + layer_bias;
        ac_layer[ind][0] = tanhM(layer[ind][0]);
        embedding[ind][0] = F * layer[ind][0];
        ac_embedding[ind][0] = tanhM(embedding[ind][0]);
        for(int i=1;i<=2;i++){
            layer[ind][i] = (W * input[ind][i]) + (H * layer[ind][i-1]) + layer_bias;
            ac_layer[ind][i] = tanhM(layer[ind][i]);
            embedding[ind][i] = F*layer[ind][i];
            ac_embedding[ind][i] = tanhM(embedding[ind][i]);
        }
    }

    void calcError(){
        for(int i=0;i<3;i++) embed_dist[i] = norm(ac_embedding[0][i]-ac_embedding[1][i]);
        for(int i=0;i<3;i++){
            string a[2] = {string(len,'0'), string(len,'0')};
            for(int j=0;j<=i;j++){
                for(int k=0;k<len;k++){
                    if(abs(input[0][j](k,0) - 1.)<1e-2) a[0][k] = '1'+j;
                    if(abs(input[1][j](k,0) - 1.)<1e-2) a[1][k] = '1'+j;
                }
            }
            edit_dist[i] = levenshtein(a[0],a[1]);
        }
    }

    void clear_gradients(){
        grad_errF *= momentum;
        grad_errH *= momentum;
        grad_errW *= momentum;
        grad_bias *= momentum;
    }
    
    void set_learning_rate(){
        default_random_engine gen;
        uniform_real_distribution<double> distrib(0.04, 0.06);
        learning_rate = distrib(gen);
    }

    void backPropagate(){
        calcError();

        Matrix grad_LiLi_1[2];
        for(int i=0;i<2;i++){
            vector<double> dg;
            for(int j=0;j<layer_dim;j++){
                dg.pb(deriv(layer[0][i+1](j,0)));
            }
            grad_LiLi_1[i] = mulDiag(H.transpose(), dg);
        }

        for(int k=2;k<3;k++){
            if(abs(embed_dist[k])<1e-2) return;
            Matrix E1_E2 = ac_embedding[0][k]-ac_embedding[1][k];
            double diff = embed_dist[k]*scale-edit_dist[k];
            Matrix grad_err_lay_bias[k+1];
            Matrix grad_errL[k+1];
            vector<double> diagE;
            for(int i=0;i<dim;i++) diagE.pb(deriv(embedding[0][k](i,0)));
            grad_errL[k] = (mulDiag(F.transpose(),diagE) * E1_E2)*(2.*diff*scale/embed_dist[k]);
            for(int i=k-1;i>=0;i--){
                grad_errL[i] = grad_LiLi_1[i] * grad_errL[i+1];
            }
            for(int i=0;i<k+1;i++){
                vector<double> dg;
                for(int j=0;j<layer_dim;j++) dg.pb(deriv(layer[0][i](j,0)));
                Matrix tmp = Diagmul(dg, grad_errL[i]);
                grad_bias += (1. - momentum) * (tmp*learning_rate);
                grad_errW += (1. - momentum) * (tmp * input[0][i].transpose());
                if(i>0) grad_errH +=  (1. - momentum) * (tmp * ac_layer[0][i-1].transpose());
            }
            grad_errF += (1. - momentum) * (Diagmul(diagE,E1_E2 * (2.*diff*scale/embed_dist[k])) *ac_layer[0][k].transpose());
        }

        for(int i=0;i<2;i++){
            vector<double> dg;
            for(int j=0;j<layer_dim;j++){
                dg.pb(deriv(layer[1][i+1](j,0)));
            }
            grad_LiLi_1[i] = mulDiag(H.transpose(), dg);
        }

        for(int k=2;k<3;k++){
            Matrix E1_E2 = ac_embedding[0][k]-ac_embedding[1][k];
            double diff = embed_dist[k]*scale-edit_dist[k];
            Matrix grad_err_lay_bias[k+1];
            Matrix grad_errL[k+1];
            vector<double> diagE;
            for(int i=0;i<dim;i++) diagE.pb(deriv(embedding[1][k](i,0)));
            grad_errL[k] = (mulDiag(F.transpose(),diagE) * E1_E2)*(2.*diff*scale/embed_dist[k]);
            for(int i=k-1;i>=0;i--){
                grad_errL[i] = grad_LiLi_1[i] * grad_errL[i+1];
            }
            for(int i=0;i<k+1;i++){
                vector<double> dg;
                for(int j=0;j<layer_dim;j++) dg.pb(deriv(layer[1][i](j,0)));
                Matrix tmp = Diagmul(dg, grad_errL[i]);
                grad_bias -= (1. - momentum) * (tmp*learning_rate);
                grad_errW -= (1. - momentum) * (tmp * input[1][i].transpose());
                if(i>0) grad_errH -= (1. - momentum) * (tmp * ac_layer[1][i-1].transpose());
            }
            grad_errF -= (1. - momentum) * (Diagmul(diagE,E1_E2 * (2.*diff*scale/embed_dist[k])) *ac_layer[1][k].transpose());   
        }
    }

    void apply_gradients(){
        /*grad_errH += Matrix::Random(H.rows(),H.cols())*0.0005;
        grad_errW += Matrix::Random(W.rows(),W.cols())*0.0005;
        grad_errF += Matrix::Random(F.rows(),F.cols())*0.0005;
        grad_bias += Matrix::Random(layer_dim,1)*0.0005;*/
        H -= grad_errH * learning_rate;
        W -= grad_errW * learning_rate;
        F -= grad_errF * learning_rate;
        layer_bias -= grad_bias * learning_rate;
    }

    void save(){
        fstream file;
        file.open(data_path, ios::out);
        file<<len<<' '<<dim<<' '<<layer_dim<<'\n';
        for(int i=0;i<W.rows();i++){
            for (int j=0;j<W.cols();j++)
                file<<W(i,j)<<' ';
            file<<'\n';
        }
        file<<'\n';
        for(int i=0;i<H.rows();i++){
            for(int j=0;j<H.cols();j++)
                file<<H(i,j)<<' ';
            file<<'\n';
        }
        file<<'\n';
        for(int i=0;i<F.rows();i++){
            for(int j=0;j<F.cols();j++)
                file<<F(i,j)<<' ';
            file<<'\n';
        }
        file<<'\n';
        for(int j=0;j<layer_dim;j++)
                file<<layer_bias(j,0)<<' ';
        file<<'\n';
        file.close();
    }

    void load_data(){
        fstream file;
        file.open(data_path, ios::in);
        if(file){
        file>>len;
        file>>dim;
        file>>layer_dim;
        W = Matrix::Zero(layer_dim, len);
        H = Matrix::Zero(layer_dim,layer_dim);
        F = Matrix::Zero(dim, layer_dim);
        for(int i=0;i<W.rows();i++)
            for(int j=0;j<W.cols();j++)
                file>>W(i,j);
        for(int i=0;i<H.rows();i++)
            for(int j=0;j<H.cols();j++)
                file>>H(i,j);
        for(int i=0;i<F.rows();i++)
            for(int j=0;j<F.cols();j++)
                file>>F(i,j);
        for(int i=0;i<layer_dim;i++)
            file>>layer_bias(i,0);
        file.close();}
    }

    string gen_rand_seq(){
        string alphabet[4] = {"A", "C", "G", "T"};
        string ret="";
        for(int i = 0; i<len; i++){
            ret += alphabet[rand()%4];
        }
        return ret;
    }

    string rand_with_dist(string a){
        int half = rand()%5;
        if(half==0) return gen_rand_seq();
        int dst = (rand()%((int)(len*0.95)))+1;
        string ret=a;
        char c[4]={'A','G','C','T'};
        for(int i=0;i<dst;i++){
            int tmp = rand()%4;
            int pos = rand()%len;
            ret[pos] = c[tmp];
        }
        return ret;
    }

    void train(){
        load_data();
        double errormean = 0;
        for(int i=1;i<=iterations;i++){
            for(int j=0;j<20000;j++){
                clear_gradients();
                set_learning_rate();
                learning_rate = learning_rate / (double)(batch_size);
                for(int k=0;k<batch_size;k++){
                    clear_input();
                    add_input(gen_rand_seq(),0);
                    add_input(rand_with_dist(input_str[0]),1);
                    execute(0);
                    execute(1);
                    backPropagate();
                }
                apply_gradients();
            }
            save();
            errormean=0;
        }
    }

    void gen_data(){
        load_data();
        freopen("data.txt","w",stdout);
        for(int i=0;i<iter;i++){
            clear_input();
            add_input(gen_rand_seq(),0);
            add_input(rand_with_dist(input_str[0]),1);
            execute(0);
            execute(1);
            calcError();
            cout<<edit_dist[2]<<' '<<embed_dist[2]*scale<<'\n';
        }
    }
};
