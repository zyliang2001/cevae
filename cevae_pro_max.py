import torch
from torch import nn
from sklearn.isotonic import IsotonicRegression

class cevae_pro_max(nn.Module):
    """
    Causal Effect Variational Autoencoder (CEVAE) model
    Expected variables:
    - x: features with some continuous and some discrete variables
    - t: treatment variable (binary)
    - y: outcome variable (binary)
    - z: latent variable (with custom dimension)
    """
    def __init__(self, x_dim, num_con_x, t_dim=1, y_dim=1, z_dim=20):
        super(cevae_pro_max, self).__init__()
        self.x_dim = x_dim
        self.num_con_x = num_con_x
        self.num_dis_x = x_dim - num_con_x
        self.t_dim = t_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.z_cov_dim = int((self.z_dim**2-self.z_dim)/2+self.z_dim)

        self.__build_encoder()
        self.__build_decoder()

    def __reparam_gaussian(self, mu, var):
        eps = torch.randn_like(var)
        return mu + eps*torch.sqrt(var)
    
    def __reparam_gaussian_with_cov(self, mu, cov):
        L = torch.linalg.cholesky(cov)
        eps = torch.randn(len(mu), self.z_dim, device=mu.device)
        sampled_mu = torch.empty((0, self.z_dim), device=mu.device)
        for i in range(len(mu)):
            sampled_mu = torch.cat((sampled_mu, (mu[i] + torch.matmul(eps[i], L[i].transpose(-2, -1))).unsqueeze(0)), dim=0)
        return sampled_mu

    def __build_encoder(self):
        # p(t|x)
        self.t_encoder = nn.Sequential(
            nn.Linear(self.x_dim, 200),
            nn.ELU(),
            nn.Linear(200, 1),
            nn.Sigmoid() # not in original code but needed for probability output
        )

        # p(y|x, t)
        self.y_encoder = nn.Sequential(
            nn.Linear(self.x_dim, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU()
        )
        self.y_1_encoder = nn.Sequential(
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200, self.y_dim),
            nn.Sigmoid()
        )
        self.y_0_encoder = nn.Sequential(
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200, self.y_dim),
            nn.Sigmoid()
        )

        # p(z|x, y, t)
        # mu is not bounded but logvar is bounded in (0, inf)
        self.z_encoder = nn.Sequential(
            nn.Linear(self.x_dim + self.y_dim, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU()
        )
        self.z_1_encoder = nn.Sequential(
            nn.Linear(200, 200),
            nn.ELU()
        )
        self.z_1_mu_encoder = nn.Sequential(
            nn.Linear(200, self.z_dim)
        )
        self.z_1_var_encoder = nn.Sequential(
            nn.Linear(200, self.z_dim**2),
            nn.Softplus()
        ) 

        self.z_0_encoder = nn.Sequential(
            nn.Linear(200, 200),
            nn.ELU()
        )
        self.z_0_mu_encoder = nn.Sequential(
            nn.Linear(200, self.z_dim)
        )
        self.z_0_var_encoder = nn.Sequential(
            nn.Linear(200, self.z_dim**2),
            nn.Softplus()
        )
    
    def __build_decoder(self):
        # p(t|z)
        self.t_decoder = nn.Sequential(
            nn.Linear(self.z_dim, 200),
            nn.ELU(),
            nn.Linear(200, self.t_dim),
            nn.Sigmoid() # not in original code but needed for probability output
        )

        # p(x_con|z)
        # mu is not bounded but var is bounded in (0, inf)
        self.x_decoder = nn.Sequential(
            nn.Linear(self.z_dim, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU()
        )
        self.x_con_decoder = nn.Sequential(
            nn.Linear(200, 200),
            nn.ELU(),
        )
        self.x_con_mu_decoder = nn.Sequential(
            nn.Linear(200, self.num_con_x),
        )
        self.x_con_var_decoder = nn.Sequential(
            nn.Linear(200, self.num_con_x),
            nn.Softplus()
        )

        # p(x_dis|z)
        self.x_dis_decoder = nn.Sequential(
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200, self.num_dis_x),
            nn.Sigmoid() # not in original code but needed for probability output
        )

        # p(y|z, t)
        # mu is not bounded and var is fixed to 1
        self.y_1_decoder = nn.Sequential(
            nn.Linear(self.z_dim, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200, self.y_dim),
            nn.Sigmoid() # not in original code but needed for probability output
        )
        self.y_0_decoder = nn.Sequential(
            nn.Linear(self.z_dim, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200, self.y_dim),
            nn.Sigmoid() # not in original code but needed for probability output
        )

    def forward(self, batch):
        """
        forward pass with input x alone
        """
        x = batch['x']
        t_hat_enc = batch['t'].unsqueeze(1)
        t_hat_dec = batch['t'].unsqueeze(1)

        # Encoder network
        # p(t|x)
        q_t = self.t_encoder(x)
        # q_t_reshaped = torch.cat((q_t, 1 - q_t), dim=1)
        # log_q_t_reshaped = torch.log(q_t_reshaped)
        # t_hat_enc = torch.nn.functional.gumbel_softmax(log_q_t_reshaped, tau=1, hard=True, dim=-1)[:, 0].unsqueeze(1) # reparametrization

        # p(y|x, t)
        q_y_temp = self.y_encoder(x)
        q_y_1 = self.y_1_encoder(q_y_temp)
        q_y_0 = self.y_0_encoder(q_y_temp)
        q_y = q_y_1 * t_hat_enc + q_y_0 * (1 - t_hat_enc) # select y based on predicted t
        q_y_reshaped = torch.cat((q_y, 1 - q_y), dim=1)
        log_q_y_reshaped = torch.log(q_y_reshaped)
        y_hat_enc = torch.nn.functional.gumbel_softmax(log_q_y_reshaped, tau=1, hard=True, dim=-1)[:, 0].unsqueeze(1) # reparametrization

        # p(z|x, y, t)
        x_y_concat = torch.cat([x, y_hat_enc], dim=1)
        z_temp = self.z_encoder(x_y_concat)
        z_1_temp = self.z_1_encoder(z_temp)
        z_1_mu = self.z_1_mu_encoder(z_1_temp)
        z_1_var = self.z_1_var_encoder(z_1_temp)
        z_0_temp = self.z_0_encoder(z_temp)
        z_0_mu = self.z_0_mu_encoder(z_0_temp)
        z_0_var = self.z_0_var_encoder(z_0_temp)
        z_mu = z_1_mu * t_hat_enc + z_0_mu * (1 - t_hat_enc) # select z mu based on predicted t
        z_covar_temp = z_1_var * t_hat_enc + z_0_var * (1 - t_hat_enc) # select z var based on predicted t
        z_covar_temp = z_covar_temp.reshape(-1, self.z_dim, self.z_dim)
        z_covar = torch.matmul(z_covar_temp, z_covar_temp.transpose(-2, -1))
        z_covar += torch.eye(self.z_dim, device=z_covar.device) * 0.1
        z = self.__reparam_gaussian_with_cov(z_mu, z_covar) # reparametrization

        # Decoder network
        # p(t|z)
        p_t = self.t_decoder(z)
        # p_t_reshaped = torch.cat((p_t, 1 - p_t), dim=1)
        # log_p_t_reshaped = torch.log(p_t_reshaped)
        # t_hat_dec = torch.nn.functional.gumbel_softmax(log_p_t_reshaped, tau=1, hard=True, dim=-1)[:, 0].unsqueeze(1) # reparametrization
        
        x_temp = self.x_decoder(z)

        # p(x_con|z)
        x_con_temp = self.x_con_decoder(x_temp)
        x_con_mu = self.x_con_mu_decoder(x_con_temp)
        x_con_var = self.x_con_var_decoder(x_con_temp)

        # p(x_dis|z)
        p_x_dis = self.x_dis_decoder(x_temp)
        
        # p(y|z, t)
        p_y_1 = self.y_1_decoder(z)
        p_y_0 = self.y_0_decoder(z)
        p_y = p_y_1 * t_hat_dec + p_y_0 * (1 - t_hat_dec)
        p_y_reshaped = torch.cat((p_y, 1 - p_y), dim=1)
        log_p_y_reshaped = torch.log(p_y_reshaped)
        y_hat_dec = torch.nn.functional.gumbel_softmax(log_p_y_reshaped, tau=1, hard=True, dim=-1)[:, 0].unsqueeze(1) # reparametrization
        
        return q_t, q_y, z_mu, z_covar, p_t, x_con_mu, x_con_var, p_x_dis, p_y, y_hat_dec
    
    def inference(self, batch):
    #     """
    #     predicting with input x and pre-determined t
    #     """
    #     x = batch['x']
    #     t = batch['t'].unsqueeze(1)
    #     t_hat_enc = t
    #     t_hat_dec = t

    #     # Encoder network
    #     # p(t|z) is voided

    #     # p(y|x, t)
    #     q_y_temp = self.y_encoder(x)
    #     q_y_1 = self.y_1_encoder(q_y_temp)
    #     q_y_0 = self.y_0_encoder(q_y_temp)
    #     q_y = q_y_1 * t_hat_enc + q_y_0 * (1 - t_hat_enc) # select y based on predicted t
    #     q_y_reshaped = torch.cat((q_y, 1 - q_y), dim=1)
    #     log_q_y_reshaped = torch.log(q_y_reshaped)
    #     y_hat_enc = torch.nn.functional.gumbel_softmax(log_q_y_reshaped, tau=1, hard=True, dim=-1)[:, 0].unsqueeze(1) # reparametrization
        
    #     # p(z|x, y, t)
    #     x_y_concat = torch.cat([x, y_hat_enc], dim=1)
    #     z_temp = self.z_encoder(x_y_concat)
    #     z_1_temp = self.z_1_encoder(z_temp)
    #     z_1_mu = self.z_1_mu_encoder(z_1_temp)
    #     z_1_var = self.z_1_var_encoder(z_1_temp)
    #     z_0_temp = self.z_0_encoder(z_temp)
    #     z_0_mu = self.z_0_mu_encoder(z_0_temp)
    #     z_0_var = self.z_0_var_encoder(z_0_temp)
    #     z_mu = z_1_mu * t_hat_enc + z_0_mu * (1 - t_hat_enc) # select z mu based on predicted t
    #     z_var = z_1_var * t_hat_enc + z_0_var * (1 - t_hat_enc) # select z var based on predicted t
    #     z = self.__reparam_gaussian(z_mu, z_var) # reparametrization

    #     # Decoder network
    #     # p(t|z) is voided
        
    #     x_temp = self.x_decoder(z)

    #     # p(x_con|z)
    #     x_con_temp = self.x_con_decoder(x_temp)
    #     x_con_mu = self.x_con_mu_decoder(x_con_temp)
    #     x_con_var = self.x_con_var_decoder(x_con_temp)

    #     # p(x_dis|z)
    #     p_x_dis = self.x_dis_decoder(x_temp)
        
    #     # p(y|z, t)
    #     p_y_1 = self.y_1_decoder(z)
    #     p_y_0 = self.y_0_decoder(z)
    #     p_y = p_y_1 * t_hat_dec + p_y_0 * (1 - t_hat_dec)
    #     p_y_reshaped = torch.cat((p_y, 1 - p_y), dim=1)
    #     log_p_y_reshaped = torch.log(p_y_reshaped)
    #     y_hat_dec = torch.nn.functional.gumbel_softmax(log_p_y_reshaped, tau=1, hard=True, dim=-1)[:, 0].unsqueeze(1) # reparametrization
        
    #     return q_y, z_mu, z_var, x_con_mu, x_con_var, p_x_dis, p_y_1, p_y_0, p_y, y_hat_dec
        return NotImplementedError
    
    def cate(self, batch):
        """
        predicting CATE with input x and pre-determined t
        """
        inference_0_sample = batch.copy()
        inference_0_sample['t'] = torch.zeros_like(inference_0_sample['t'])
        inference_0 = self.forward(inference_0_sample)

        inference_1_sample = batch.copy()
        inference_1_sample['t'] = torch.ones_like(inference_1_sample['t'])
        inference_1 = self.forward(inference_1_sample)

        return torch.bernoulli(inference_1[-2]) - torch.bernoulli(inference_0[-2])

    def bern_prob_loss(self, pred_p, true_value):
        prob_loss_1 = torch.log(pred_p + 1e-10)
        prob_loss_0 = torch.log(1 - pred_p + 1e-10)
        return -torch.mean(true_value * prob_loss_1 + (1 - true_value) * prob_loss_0)
    
    def gaus_prob_loss(self, pred_mu, pred_var, true_value):
        dist = torch.distributions.Normal(pred_mu, torch.sqrt(pred_var))
        return -torch.mean(dist.log_prob(true_value))
    
    
    def kl_divergence_cov(self, pred_mu, pred_cov, prior_mu=0, prior_cov=1):
        """
        Calculate the KL divergence between two multivariate Gaussian distributions.
        
        Parameters:
            pred_mu (torch.Tensor): Mean vector of the first distribution.
            pred_cov (torch.Tensor): Covariance matrix of the first distribution.
            prior_mu (torch.Tensor): Mean vector of the second distribution.
            prior_cov (torch.Tensor): Covariance matrix of the second distribution.
            
        Returns:
            kl_divergence (torch.Tensor): KL divergence between the two distributions.
        """
        prior_cov = torch.empty((0, self.z_dim, self.z_dim), device=pred_cov.device)
        for _ in range(pred_cov.size(0)):
            prior_cov = torch.cat((prior_cov, torch.eye(self.z_dim, device=pred_cov.device).unsqueeze(0)), dim=0)
        k = pred_mu.size(1)  # Dimensionality of the distributions
        prior_cov_inv = torch.inverse(prior_cov)
        
        term1_vec = []
        term2_vec = []
        termk_vec = torch.tensor(k).repeat(pred_mu.size(0),)
        term4_vec = []

        for i in range(pred_cov.size(0)):
            term1_vec.append(torch.trace(torch.matmul(prior_cov_inv[i], pred_cov[i])))
            term2_vec.append(torch.matmul(torch.matmul((prior_mu - pred_mu)[i].T, prior_cov_inv[i]), (prior_mu - pred_mu)[i]))
            term4_vec.append(torch.log(torch.det(prior_cov[i]) / torch.det(pred_cov[i])))
        
        kl_divergence = 0.5 * (torch.tensor(term1_vec) + torch.tensor(term2_vec) - termk_vec + torch.tensor(term4_vec))
        
        return torch.mean(kl_divergence)

    def pdist2sq(self, X, Y):
        """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
        C = -2 * torch.matmul(X, Y.t())
        nx = torch.sum(X**2, dim=1, keepdim=True)
        ny = torch.sum(Y**2, dim=1, keepdim=True)
        D = (C + ny.t()) + nx
        return torch.sqrt(torch.clamp(D, min=1e-10))

    def mmd_loss(self, z_representation, treatment, proportion_treated):
        it = torch.where(treatment > 0)[0]
        ic = torch.where(treatment < 1)[0]

        Xc = z_representation[ic]
        Xt = z_representation[it]

        sig = 1
        Kcc = torch.exp(-self.pdist2sq(Xc, Xc) / sig**2)
        Kct = torch.exp(-self.pdist2sq(Xc, Xt) / sig**2)
        Ktt = torch.exp(-self.pdist2sq(Xt, Xt) / sig**2)

        m = float(Xc.shape[0])
        n = float(Xt.shape[0])

        mmd = ((1.0 - proportion_treated)**2 / (m * (m - 1.0))) * (torch.sum(Kcc) - m)
        mmd = mmd + (proportion_treated**2 / (n * (n - 1.0))) * (torch.sum(Ktt) - n)
        mmd = mmd - 2.0 * proportion_treated * (1.0 - proportion_treated) / (m * n) * torch.sum(Kct)
        mmd = 4.0 * mmd
        
        return mmd
        
    def train_loss(self, output, batch):
        q_t, q_y, z_mu, z_var, p_t, x_con_mu, x_con_var, p_x_dis, p_y, y_hat_dec = output
        q_t_loss = self.bern_prob_loss(q_t, batch['t'].view(-1,1))
        q_y_loss = 10*self.bern_prob_loss(q_y, batch['y'].view(-1,1))
        z_kl_loss = 0.1*self.kl_divergence_cov(z_mu, z_var)
        mmd_loss = self.mmd_loss(z_mu, batch['t'], batch['t'].mean())
        p_t_loss = self.bern_prob_loss(p_t, batch['t'].view(-1,1))
        p_x_con_loss = 0.01*self.gaus_prob_loss(x_con_mu, x_con_var, batch['x'][:, -1*self.num_con_x:])
        p_x_dis_loss = self.bern_prob_loss(p_x_dis, batch['x'][:, :-1*self.num_con_x])
        p_y_loss =  10*self.bern_prob_loss(p_y, batch['y'].view(-1,1)) 
        loss = q_t_loss + q_y_loss + z_kl_loss + mmd_loss + p_t_loss + p_x_con_loss + p_x_dis_loss + p_y_loss
        return loss, q_t_loss, q_y_loss, z_kl_loss, mmd_loss, p_t_loss, p_x_con_loss, p_x_dis_loss, p_y_loss
    
    def inference_loss(self, output, batch):
        q_y, z_mu, z_var, x_con_mu, x_con_var, p_x_dis, p_y_1, p_y_0, p_y, y_hat_dec = output
        loss = 0
        loss += self.bern_prob_loss(q_y, batch['y'].view(-1,1)) 
        loss += self.kl_divergence(z_mu, z_var)
        loss += self.gaus_prob_loss(x_con_mu, x_con_var, batch['x'][:, -1*self.num_con_x:])
        loss += self.bern_prob_loss(p_x_dis, batch['x'][:, :-1*self.num_con_x])
        loss += self.bern_prob_loss(q_y, batch['y'].view(-1,1)) 
        return loss