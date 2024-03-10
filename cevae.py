import torch
from torch import nn

class cevae(nn.Module):
    """
    Causal Effect Variational Autoencoder (CEVAE) model
    Expected variables:
    - x: features with some continuous and some discrete variables
    - t: treatment variable (binary)
    - y: outcome variable (binary)
    - z: latent variable (with custom dimension)
    """
    def __init__(self, x_dim, num_con_x, t_dim=1, y_dim=1, z_dim=20):
        super(cevae, self).__init__()
        self.x_dim = x_dim
        self.num_con_x = num_con_x
        self.num_dis_x = x_dim - num_con_x
        self.t_dim = t_dim
        self.y_dim = y_dim
        self.z_dim = z_dim

        self.__build_encoder()
        self.__build_decoder()

    def __reparam_gaussian(self, mu, var):
        eps = torch.randn_like(var)
        return mu + eps*torch.sqrt(var)

    def __build_encoder(self):
        # p(t|x)
        t_encoder = nn.Sequential(
            nn.Linear(self.x_dim, 200),
            nn.ELU(),
            nn.Linear(200, 1),
            nn.Sigmoid() # not in original code but needed for probability output
        )

        # p(y|x, t)
        y_encoder = nn.Sequential(
            nn.Linear(self.x_dim, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU()
        )
        y_1_encoder = nn.Sequential(
            y_encoder,
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200, self.y_dim),
            nn.Sigmoid()
        )
        y_0_encoder = nn.Sequential(
            y_encoder,
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200, self.y_dim),
            nn.Sigmoid()
        )

        # p(z|x, y, t)
        # mu is not bounded but logvar is bounded in (0, inf)
        z_encoder = nn.Sequential(
            nn.Linear(self.x_dim + self.y_dim, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU()
        )
        z_1_encoder = nn.Sequential(
            z_encoder,
            nn.Linear(200, 200),
            nn.ELU()
        )
        z_1_mu_encoder = nn.Sequential(
            z_1_encoder,
            nn.Linear(200, self.z_dim)
        )
        z_1_var_encoder = nn.Sequential(
            z_1_encoder,
            nn.Linear(200, self.z_dim),
            nn.Softplus()
        )

        z_0_encoder = nn.Sequential(
            z_encoder,
            nn.Linear(200, 200),
            nn.ELU()
        )
        z_0_mu_encoder = nn.Sequential(
            z_0_encoder,
            nn.Linear(200, self.z_dim)
        )
        z_0_var_encoder = nn.Sequential(
            z_0_encoder,
            nn.Linear(200, self.z_dim),
            nn.Softplus()
        )

        self.encoder_dict = nn.ModuleDict({
            't_encoder': t_encoder,
            'y_1_encoder': y_1_encoder,
            'y_0_encoder': y_0_encoder,
            'Z_1_mu_encoder': z_1_mu_encoder,
            'Z_1_var_encoder': z_1_var_encoder,
            'Z_0_mu_encoder': z_0_mu_encoder,
            'Z_0_var_encoder': z_0_var_encoder
        })
        
    
    def __build_decoder(self):
        # p(t|z)
        t_decoder = nn.Sequential(
            nn.Linear(self.z_dim, 200),
            nn.ELU(),
            nn.Linear(200, self.t_dim),
            nn.Sigmoid() # not in original code but needed for probability output
        )

        # p(x_con|z)
        # mu is not bounded but var is bounded in (0, inf)
        x_decoder = nn.Sequential(
            nn.Linear(self.z_dim, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU()
        )
        x_con_decoder = nn.Sequential(
            x_decoder,
            nn.Linear(200, 200),
            nn.ELU(),
        )
        x_con_mu_decoder = nn.Sequential(
            x_con_decoder,
            nn.Linear(200, self.num_con_x),
        )
        x_con_var_decoder = nn.Sequential(
            x_con_decoder,
            nn.Linear(200, self.num_con_x),
            nn.Softplus()
        )

        # p(x_dis|z)
        x_dis_decoder = nn.Sequential(
            x_decoder,
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200, self.num_dis_x),
            nn.Sigmoid() # not in original code but needed for probability output
        )

        # p(y|z, t)
        # mu is not bounded and var is fixed to 1
        y_1_decoder = nn.Sequential(
            nn.Linear(self.z_dim, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200, self.y_dim),
            nn.Sigmoid() # not in original code but needed for probability output
        )
        y_0_decoder = nn.Sequential(
            nn.Linear(self.z_dim, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200, self.y_dim),
            nn.Sigmoid() # not in original code but needed for probability output
        )

        self.decoder_dict = nn.ModuleDict({
            't_decoder': t_decoder,
            'x_con_mu_decoder': x_con_mu_decoder,
            'x_con_var_decoder': x_con_var_decoder,
            'x_dis_decoder': x_dis_decoder,
            'y_1_decoder': y_1_decoder,
            'y_0_decoder': y_0_decoder
        })

    def forward(self, batch):
        """
        forward pass with input x alone
        """
        x = batch['x']

        # Encoder network
        # p(t|x)
        q_t = self.encoder_dict['t_encoder'](x) 
        t_hat_enc = nn.F.gumbel_softmax(q_t, tau=1, hard=False) # reparametrization

        # p(y|x, t)
        q_y_1 = self.encoder_dict['y_1_encoder'](x)
        q_y_0 = self.encoder_dict['y_0_encoder'](x)
        q_y = q_y_1 * t_hat_enc + q_y_0 * (1 - t_hat_enc) # select y based on predicted t
        y_hat_enc = nn.F.gumbel_softmax(q_y, tau=1, hard=False) # reparametrization

        # p(z|x, y, t)
        x_y_concat = torch.cat([x, y_hat_enc], dim=1)
        z_1_mu = self.encoder_dict['Z_1_mu_encoder'](x_y_concat)
        z_1_var = self.encoder_dict['Z_1_var_encoder'](x_y_concat)
        z_0_mu = self.encoder_dict['Z_0_mu_encoder'](x_y_concat)
        z_0_var = self.encoder_dict['Z_0_var_encoder'](x_y_concat)
        z_mu = z_1_mu * t_hat_enc + z_0_mu * (1 - t_hat_enc) # select z mu based on predicted t
        z_var = z_1_var * t_hat_enc + z_0_var * (1 - t_hat_enc) # select z var based on predicted t
        z = self.__reparam_gaussian(z_mu, z_var) # reparametrization

        # Decoder network
        # p(t|z)
        p_t = self.decoder_dict['t_decoder'](z)
        t_hat_dec = nn.F.gumbel_softmax(p_t, tau=1, hard=False)
        
        # p(x_con|z)
        x_con_mu = self.decoder_dict['x_con_mu_decoder'](z)
        x_con_var = self.decoder_dict['x_con_var_decoder'](z)

        # p(x_dis|z)
        p_x_dis = self.decoder_dict['x_dis_decoder'](z)
        
        # p(y|z, t)
        p_y_1 = self.decoder_dict['y_1_decoder'](z)
        p_y_0 = self.decoder_dict['y_0_decoder'](z)
        p_y = p_y_1 * t_hat_dec + p_y_0 * (1 - t_hat_dec)
        y_hat_dec = torch.bernoulli(p_y)
        
        return q_t, q_y, z_mu, z_var, p_t, x_con_mu, x_con_var, p_x_dis, p_y, y_hat_dec
    
    def inference(self, batch):
        """
        predicting with input x and pre-determined t
        """
        x = batch['x']
        t = batch['t']
        t_hat_enc = t
        t_hat_dec = t

        # Encoder network
        # p(t|z) is voided

        # p(y|x, t)
        q_y_1 = self.encoder_dict['y_1_encoder'](x)
        q_y_0 = self.encoder_dict['y_0_encoder'](x)
        q_y = q_y_1 * t_hat_enc + q_y_0 * (1 - t_hat_enc) # select y based on predicted t
        y_hat_enc = nn.F.gumbel_softmax(q_y, tau=1, hard=False) # reparametrization

        # p(z|x, y, t)
        x_y_concat = torch.cat([x, y_hat_enc], dim=1)
        z_1_mu = self.encoder_dict['Z_1_mu_encoder'](x_y_concat)
        z_1_var = self.encoder_dict['Z_1_var_encoder'](x_y_concat)
        z_0_mu = self.encoder_dict['Z_0_mu_encoder'](x_y_concat)
        z_0_var = self.encoder_dict['Z_0_var_encoder'](x_y_concat)
        z_mu = z_1_mu * t_hat_enc + z_0_mu * (1 - t_hat_enc) # select z mu based on predicted t
        z_var = z_1_var * t_hat_enc + z_0_var * (1 - t_hat_enc) # select z var based on predicted t
        z = self.__reparam_gaussian(z_mu, z_var) # reparametrization

        # Decoder network
        # p(t|z) is voided
        
        # p(x_con|z)
        x_con_mu = self.decoder_dict['x_con_mu_decoder'](z)
        x_con_var = self.decoder_dict['x_con_var_decoder'](z)

        # p(x_dis|z)
        p_x_dis = self.decoder_dict['x_dis_decoder'](z)
        
        # p(y|z, t)
        p_y_1 = self.decoder_dict['y_1_decoder'](z)
        p_y_0 = self.decoder_dict['y_0_decoder'](z)
        p_y = p_y_1 * t_hat_dec + p_y_0 * (1 - t_hat_dec)
        y_hat_dec = torch.bernoulli(p_y)
        
        return q_y, z_mu, z_var, x_con_mu, x_con_var, p_x_dis, p_y, y_hat_dec
    
    def cate(self, batch):
        """
        predicting CATE with input x and pre-determined t
        """
        batch['t'] = torch.zeros_like(batch['t'])
        inference_0 = self.inference(batch)
        batch['t'] = torch.ones_like(batch['t'])
        inference_1 = self.inference(batch)

        return inference_1[-1] - inference_0[-1]

    def bern_prob_loss(self, pred_p, true_value):
        prob_loss_1 = torch.log(pred_p + 1e-10)
        prob_loss_0 = torch.log(1 - pred_p + 1e-10)
        return -torch.sum(true_value * prob_loss_1 + (1 - true_value) * prob_loss_0)
    
    def gaus_prob_loss(self, pred_mu, pred_var, true_value):
        dist = torch.distributions.Normal(pred_mu, pred_var)
        return -dist.log_prob(true_value)
    
    def kl_divergence(self, pred_mus, pred_vars, prior_mu=0, prior_var=1):
        kl_divergence = 0.5 * (torch.log(prior_var / pred_vars) + (pred_vars + (pred_mus - prior_mu)**2) / prior_var - 1)
        return torch.sum(kl_divergence)