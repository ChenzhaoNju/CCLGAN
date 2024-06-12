import itertools
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
from util.image_pool import ImagePool
import torch.nn as nn
from torchvision.models.vgg import vgg16
import torchvision.transforms as transform
from torchvision import transforms
import torch.nn.functional as F
import scipy.stats as st
from segment_anything import  sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import clip
import numpy as np
from PIL import Image


class CCLModel(BaseModel):
    """ This class implements DCLGAN model.
    This code is inspired by CUT and CycleGAN.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--CCL_mode', type=str, default="CCL", choices='CCL')
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=2.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--lambda_IDT', type=float, default=1.0, help='weight for l1 identical loss: (G(X),X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False,
                            help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='4,8,12', help='compute NCE loss on which layers')
        parser.add_argument('--adv_nce_layers', type=str, default='3,7', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'],
                            help='how to downsample the feature map')
        # parser.add_argument('--netFAdvBack', type=str, default='non_localOne',
        # choices=['mlp_sample', 'non_localOne'])
        parser.add_argument('--netFAdvBack', type=str, default='mlp_sample',
                            choices=['mlp_sample', 'non_localOne'])
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization.")

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for DCLGAN.
        if opt.DCL_mode.lower() == "dcl":
            parser.set_defaults(nce_idt=True, lambda_NCE=2.0)
        else:
            raise ValueError(opt.DCL_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'NCE1', 'D_B', 'G_B', 'NCE2', 'G']
        visual_names_A = ['real_A', 'fake_B']
        visual_names_B = ['real_B', 'fake_A']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
        self.adv_nce_layers = [int(i) for i in self.opt.adv_nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['idt_B', 'idt_A']
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B

        if self.isTrain:
            self.model_names = ['G_A', 'F1', 'D_A', 'G_B', 'F2', 'D_B']
        else:  # during test time, only load G
            self.model_names = ['G_A', 'G_B']

        # define networks (both generator and discriminator)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias,
                                        opt.no_antialias_up, self.gpu_ids, opt)
        self.netG_B = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias,
                                        opt.no_antialias_up, self.gpu_ids, opt)
        self.netF1 = networks.define_F(opt.input_nc, opt.netF, opt.normG,
                                       not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids,
                                       opt)
        self.netF2 = networks.define_F(opt.input_nc, opt.netF, opt.normG,
                                       not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids,
                                       opt)
        self.netAdvBackSample = networks.define_F(opt.input_nc, opt.netFAdvBack, opt.normG, not opt.no_dropout,
                                                  opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        self.netAdvrainSample = networks.define_F(opt.input_nc, opt.netFAdvBack, opt.normG, not opt.no_dropout,
                                                  opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        self.vgg16=Vgg16()
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.text = clip.tokenize(["rain image", "rain-free image"]).to(self.device)
        self.vgg166 =vgg16(torch.load("vgg16-397923af.pth")).features.cuda()


        if self.isTrain:
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias,
                                            self.gpu_ids, opt)
            self.netD_B = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias,
                                            self.gpu_ids, opt)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []
            self.l1 = nn.L1Loss()
            self.l2 = nn.MSELoss(reduction='mean')
            self.cos = torch.nn.CosineSimilarity(dim=-1)
            self.col = torch.nn.MSELoss()
            self.blur = Blur(3).cuda()

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.criterionSim = torch.nn.L1Loss('sum').to(self.device)
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()  # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_G_loss().backward()  # calculate graidents for G
            self.backward_D_A()  # calculate gradients for D_A
            self.backward_D_B()  # calculate graidents for D_B
            self.optimizer_F = torch.optim.Adam(itertools.chain(self.netF1.parameters(), self.netF2.parameters()))
            self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self,epoch):
        # forward
        self.forward()

        # update D
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()  # calculate gradients for D_A
        self.backward_D_B()  # calculate graidents for D_B
        self.optimizer_D.step()

        # update G
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss(epoch)
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)

        if self.opt.nce_idt:
            self.idt_A = self.netG_A(self.real_B)
            self.idt_B = self.netG_B(self.real_A)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B) * self.opt.lambda_GAN

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A) * self.opt.lambda_GAN

    def compute_G_loss(self,epoch):
        """Calculate GAN and NCE loss for the generator"""
        fakeB = self.fake_B
        fakeA = self.fake_A

        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fakeB = self.netD_A(fakeB)
            pred_fakeA = self.netD_B(fakeA)
            self.loss_G_A = self.criterionGAN(pred_fakeB, True).mean() * self.opt.lambda_GAN
            self.loss_G_B = self.criterionGAN(pred_fakeA, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_A = 0.0
            self.loss_G_B = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE1 = self.calculate_NCE_loss1(self.real_A, self.fake_B) * self.opt.lambda_NCE
            self.loss_NCE2 = self.calculate_NCE_loss2(self.real_B, self.fake_A) * self.opt.lambda_NCE
            self.loss_NCE3 = self.calculate_NCE_loss3(self.fake_B, self.rec_A) * self.opt.lambda_NCE
            self.loss_NCE4 = self.calculate_NCE_loss4(self.fake_A, self.rec_B) * self.opt.lambda_NCE
        else:
            self.loss_NCE1, self.loss_NCE_bd, self.loss_NCE2 = 0.0, 0.0, 0.0
        if self.opt.lambda_NCE > 0.0:

            # L1 IDENTICAL Loss
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * self.opt.lambda_IDT
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * self.opt.lambda_IDT
            loss_NCE_both = (self.loss_NCE1 + self.loss_NCE2 + self.loss_NCE3 + self.loss_NCE4) * 0.25 + (
                    self.loss_idt_A + self.loss_idt_B) * 0.5

        else:
            loss_NCE_both = (self.loss_NCE1 + self.loss_NCE2) * 0.5

        loss_cyc_NCE1 = self.calculate_wai_cyc_NCE_loss1(self.real_B, self.rec_A, self.fake_B, self.real_A)+self.calculate_wai_cyc_NCER1(self.real_B, self.rec_A, self.fake_B, self.real_A)
        loss_cyc_NCE2 = self.calculate_wai_cyc_NCE_loss2(self.real_A, self.rec_B, self.fake_A, self.real_B)+self.calculate_wai_cyc_NCER2(self.real_A, self.rec_B, self.fake_A, self.real_B)

        loss_ncyc_NCE1 = self.calculate_DisNCE_loss1(self.real_A, self.fake_B, self.rec_B, self.real_B)
        loss_ncyc_NCE2 = self.calculate_DisNCE_loss2(self.real_A, self.fake_A, self.rec_A, self.real_B)
        
        #loss_col1 =self.color1(self.real_A, self.rec_A)
        #loss_col2 = self.color2(self.real_B, self.rec_B)
        #loss_col=loss_col1+loss_col2

        loss_Wcyc_NCE = (loss_cyc_NCE1 + loss_cyc_NCE2) * 0.5
        loss_Ncyc_NCE = (loss_ncyc_NCE1 + loss_ncyc_NCE2) * 0.005
        if epoch > 300:
              loss_SC1=self.SAM(self.real_A, self.rec_A)
              loss_SC2=self.SAM1(self.real_B, self.rec_B)
              losssc=0.5*(loss_SC1+loss_SC2)
        
        self.loss_G = (self.loss_G_A + self.loss_G_B) * 0.5 +loss_Wcyc_NCE+loss_Ncyc_NCE+ loss_NCE_both+losssc
        return self.loss_G


    def color1(self, src, tgt):
        loss_color = self.col(self.blur(src).squeeze(-1), self.blur(tgt).squeeze(-1)) * 0.05
        return loss_color
    def color2(self, src, tgt):
        loss_color = self.col(self.blur(src).squeeze(-1), self.blur(tgt).squeeze(-1)) * 0.05
        return loss_color


    def SAM(self, a, b):
        
        a=a.squeeze(0)
        b=b.squeeze(0)
        
        sam_transform = ResizeLongestSide(self.sam.image_encoder.img_size)
        resampled_image = sam_transform.apply_image(a)
        
        resampled_image_tensor = torch.as_tensor(resampled_image.transpose(2, 0, 1)).to(self.device)
        
        resampled_image = self.sam.preprocess(resampled_image_tensor[None, :, :, :])  # (1, 3, 1024, 1024)
        assert resampled_image.shape == (1, 3, self.sam.image_encoder.img_size,
                                     self.sam.image_encoder.img_size), 'input image should be resized to 1024*1024'

        resampled_image_generated = sam_transform.apply_image(b)
        resampled_image_generated_tensor = torch.as_tensor(resampled_image_generated.transpose(2, 0, 1)).to(self.device)
        resampled_image_generated = self.sam.preprocess(resampled_image_generated_tensor[None, :, :, :])
        assert resampled_image_generated.shape == (1, 3, self.sam.image_encoder.img_size,
                                            self.sam.image_encoder.img_size), 'input image should be resized to 1024*1024'

        
        with torch.no_grad():
               embedding = self.sam.image_encoder(resampled_image)
        with torch.no_grad():
               embedding_generated = self.sam.image_encoder(resampled_image_generated)
        samscore = 1-cosine_similarity(embedding, embedding_generated)
        
        return samscore

    def SAM1(self, a, b):
       
        a=a.squeeze(0)
        b=b.squeeze(0)
       
        sam_transform = ResizeLongestSide(self.sam.image_encoder.img_size)
        resampled_image = sam_transform.apply_image(a)
       
        resampled_image_tensor = torch.as_tensor(resampled_image.transpose(2, 0, 1)).to(self.device)
        
        resampled_image = self.sam.preprocess(resampled_image_tensor[None, :, :, :])  # (1, 3, 1024, 1024)
        assert resampled_image.shape == (1, 3, self.sam.image_encoder.img_size,
                                     self.sam.image_encoder.img_size), 'input image should be resized to 1024*1024'

        resampled_image_generated = sam_transform.apply_image(b)
        resampled_image_generated_tensor = torch.as_tensor(resampled_image_generated.transpose(2, 0, 1)).to(self.device)
        resampled_image_generated = self.sam.preprocess(resampled_image_generated_tensor[None, :, :, :])
        assert resampled_image_generated.shape == (1, 3, self.sam.image_encoder.img_size,
                                            self.sam.image_encoder.img_size), 'input image should be resized to 1024*1024'

        
        with torch.no_grad():
               embedding = self.sam.image_encoder(resampled_image)
        with torch.no_grad():
               embedding_generated = self.sam.image_encoder(resampled_image_generated)
        samscore = 1-cosine_similarity(embedding, embedding_generated)
        
        return samscore
   

    def calculate_NCE_loss1(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG_B(tgt, self.nce_layers, encode_only=True)
        feat_k = self.netG_A(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF1(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF2(feat_q, self.opt.num_patches, sample_ids)
        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k)
            total_nce_loss += loss.mean()
        return total_nce_loss / n_layers

    def calculate_NCE_loss2(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG_A(tgt, self.nce_layers, encode_only=True)
        feat_k = self.netG_B(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF2(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF1(feat_q, self.opt.num_patches, sample_ids)
        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k)
            total_nce_loss += loss.mean()
        return total_nce_loss / n_layers

    def calculate_NCE_loss3(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG_B(tgt, self.nce_layers, encode_only=True)
        feat_k = self.netG_A(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF2(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF1(feat_q, self.opt.num_patches, sample_ids)
        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k)
            total_nce_loss += loss.mean()
        return total_nce_loss / n_layers

    def calculate_NCE_loss4(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG_A(tgt, self.nce_layers, encode_only=True)
        feat_k = self.netG_B(src, self.nce_layers, encode_only=True)
        # print(type(feat_k))
        feat_k_pool, sample_ids = self.netF1(feat_k, self.opt.num_patches, None)
        # print(type(feat_k_pool))
        feat_q_pool, _ = self.netF2(feat_q, self.opt.num_patches, sample_ids)
        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k)
            total_nce_loss += loss.mean()
        return total_nce_loss / n_layers

    def calculate_wai_cyc_NCE_loss1(self, src, tgt, tgt1, rtgt):

        
        tgt=transform.Resize((224,224))(tgt)
        tgt1=transform.Resize((224,224))(tgt1)
        rtgt=transform.Resize((224,224))(rtgt)
        src = transform.Resize((224, 224))(src)
        a_vgg = self.model.encode_image(tgt)
        b_vgg = self.model.encode_image(tgt1)
        p_vgg = self.model.encode_image(rtgt)
        n_vgg = self.model.encode_image(src)
        pix_loss = 0
        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())

            d_an = self.l1(a_vgg[i], n_vgg[i].detach())

            d_ab = self.l1(a_vgg[i], b_vgg[i].detach())

            contra = d_ap / (d_an + 0.1 * d_ab + 1e-7)

            pix_loss += contra
        
        logits_per_image, logits_per_text = self.model(tgt, self.text)
        probs = logits_per_image.softmax(dim=-1)
        loss_D_A_CLIP = self.criterionCycle(probs, src)

        logits_per_image, logits_per_text = self.model(tgt1, self.text)
        probs = logits_per_image.softmax(dim=-1)
        loss_D_A_CLIP += self.criterionCycle(probs, src)
        text_loss=loss_D_A_CLIP+loss_D_A_CLIP

        #dualloss = pix_loss+pixx_loss
        dualloss = pix_loss+text_loss

        return dualloss

    def calculate_wai_cyc_NCE_loss2(self, src, tgt, tgt1, rtgt):

        

        tgt = transform.Resize((224, 224))(tgt)
        tgt1 = transform.Resize((224, 224))(tgt1)
        rtgt = transform.Resize((224, 224))(rtgt)
        src = transform.Resize((224, 224))(src)
        a_vgg = self.model.encode_image(tgt)
        b_vgg = self.model.encode_image(tgt1)
        p_vgg = self.model.encode_image(rtgt)
        n_vgg = self.model.encode_image(src)
        pix_loss = 0
        weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            # print(d_ap)
            # print(type(a_vgg[i]))
            # d_bn=self.l1(b_vgg[i],n_vgg[i].detach())
            d_an = self.l1(a_vgg[i], n_vgg[i].detach())
            # d_bp=self.l1(b_vgg[i],p_vgg[i].detach())
            d_ba = self.l1(a_vgg[i], b_vgg[i].detach())
            # c1=d_ap+d_bp+0.1*d_ba
            # print(c1)
            # c2=(d_an+d_bn+1e-7)
            # contra=c1/c2
            # contra = d_ap / (0.01 * d_an + 0.01 * d_ba + 1e-7 + d_ap)
            contra = d_ap / (d_an + 0.1 * d_ba + 1e-7)
            # contra = d_ap
            # pix_loss += weights[i] * contra
            pix_loss += contra
            # pix_loss+=weights[i]*d_ap
        
        logits_per_image, logits_per_text = self.model(tgt, self.text)
        probs = logits_per_image.softmax(dim=-1)
        loss_D_A_CLIP = self.criterionCycle(probs, src)

        logits_per_image, logits_per_text = self.model(tgt1, self.text)
        probs = logits_per_image.softmax(dim=-1)
        loss_D_A_CLIP += self.criterionCycle(probs, src)
        text_loss=loss_D_A_CLIP+loss_D_A_CLIP

        dualloss = pix_loss+text_loss

        return dualloss


    def calculate_wai_cyc_NCER1(self, src, tgt, tgt1, rtgt):

        va_vgg = self.vgg16(tgt)
        
        vb_vgg = self.vgg16(tgt1)
        vp_vgg = self.vgg16(rtgt)
        vn_vgg = self.vgg16(src)
        
        pixx_loss=0
        weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        for i in range(len(va_vgg)):
            d_ap = self.l1(va_vgg[i], vp_vgg[i].detach())
            d_an = self.l1(va_vgg[i], vn_vgg[i].detach())
            d_ab = self.l1(va_vgg[i], vb_vgg[i].detach())
            contra = d_ap / (d_an + 0.1 * d_ab + 1e-7)
            pixx_loss += weights[i] *contra
            
        
        dualloss = pixx_loss
        

        return dualloss

    def calculate_wai_cyc_NCER2(self, src, tgt, tgt1, rtgt):

        va_vgg = self.vgg16(tgt)
        vb_vgg = self.vgg16(tgt1)
        vp_vgg = self.vgg16(rtgt)
        vn_vgg = self.vgg16(src)
        pixx_loss = 0
        weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        for i in range(len(va_vgg)):
            d_ap = self.l1(va_vgg[i], vp_vgg[i].detach())
            d_an = self.l1(va_vgg[i], vn_vgg[i].detach())
            d_ab = self.l1(va_vgg[i], vb_vgg[i].detach())
            contra = d_ap / (d_an + 0.1 * d_ab + 1e-7)
            pixx_loss += weights[i] * contra

        

        dualloss = pixx_loss

        return dualloss


    def calculate_tri_NCE_loss2(self, src, tgt, rtgt):
        # vgg16=nn.DataParallel(Vgg16())
        # vgg16=Vgg16()
        # a_vgg = self.vgg16(tgt)
        # p_vgg = self.vgg16(rtgt)
        # n_vgg = self.vgg16(src)

        pix_loss = 0
        weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            d_an = self.l1(a_vgg[i], n_vgg[i].detach())
            contra = d_ap / (d_an + 1e-7)
            pix_loss += weights[i] * contra
            # pix_loss+=weights[i]*d_ap

        L_const = networks.PerceptualLoss()
        # style_loss = torch.mean(max(L_const(a_vgg, p_vgg) -L_const(a_vgg,n_vgg) + 0.04 ,L_const(a_vgg, p_vgg)-L_const(a_vgg, p_vgg)))

        # dualloss=pix_loss+0.5*style_loss
        dualloss = pix_loss
        # dualloss=0.5*style_loss

        # dualloss=0.005*loss_cont

        return dualloss

    def calculate_tri_NCE_loss3(self, src, tgt, rtgt):
        # vgg16=nn.DataParallel(Vgg16())
        # vgg16=Vgg16()
        a_vgg = self.vgg16(tgt)
        p_vgg = self.vgg16(rtgt)
        n_vgg = self.vgg16(src)

        pix_loss = 0
        weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            d_an = self.l1(a_vgg[i], n_vgg[i].detach())
            contra = d_ap / (d_an + 1e-7)
            pix_loss += weights[i] * contra
            # pix_loss+=weights[i]*d_ap

        L_const = networks.PerceptualLoss()
        # style_loss = torch.mean(max(L_const(a_vgg, p_vgg) -L_const(a_vgg,n_vgg) + 0.04 ,L_const(a_vgg, p_vgg)-L_const(a_vgg, p_vgg)))

        # dualloss=pix_loss+0.5*style_loss
        dualloss = pix_loss
        # dualloss=0.5*style_loss

        # dualloss=0.005*loss_cont
        return dualloss

    def calculate_tri_NCE_loss4(self, src, tgt, rtgt):
        # vgg16=nn.DataParallel(Vgg16())
        # vgg16=Vgg16()
        a_vgg = self.vgg16(tgt)
        p_vgg = self.vgg16(rtgt)
        n_vgg = self.vgg16(src)

        pix_loss = 0
        weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            d_an = self.l1(a_vgg[i], n_vgg[i].detach())
            contra = d_ap / (d_an + 1e-7)
            pix_loss += weights[i] * contra
            # pix_loss+=weights[i]*d_ap

        L_const = networks.PerceptualLoss()
        # style_loss = torch.mean(max(L_const(a_vgg, p_vgg) -L_const(a_vgg,n_vgg) + 0.04 ,L_const(a_vgg, p_vgg)-L_const(a_vgg, p_vgg)))

        # dualloss=pix_loss+0.5*style_loss
        dualloss = pix_loss
        # dualloss=0.5*style_loss

        # dualloss=0.005*loss_cont
        return dualloss

    def calculate_DisNCE_loss1(self, src, tgt, tgt1, rtgt):

        # timestrap1 = time.time()
        feat_real_A = self.netD_A(src, self.adv_nce_layers, encode_only=True)
        feat_fake_B = self.netD_A(tgt, self.adv_nce_layers, encode_only=True)
        feat_rec_B = self.netD_A(tgt1, self.adv_nce_layers, encode_only=True)
        feat_real_B = self.netD_A(rtgt, self.adv_nce_layers, encode_only=True)

        feat_a_real_A = self.netAdvBackSample(feat_real_A)
        
        feat_a_fake_B = self.netAdvBackSample(feat_fake_B)
        feat_a_rec_B = self.netAdvBackSample(feat_rec_B)
        feat_b_pool = self.netAdvrainSample(feat_real_B)


        total_dis_loss = 0.0
        for i in range(len(feat_a_real_A)):
            for j in range(len(feat_a_real_A[i])):
                d_21 = self.l1(feat_a_fake_B[i][j].float(), feat_a_real_A[i][j].float())
                d_31 = self.l1(feat_a_rec_B[i][j].float(), feat_a_real_A[i][j].float())
                d_24 = self.l1(feat_a_fake_B[i][j].float(), feat_b_pool[i][j].float())
                d_34 = self.l1(feat_a_rec_B[i][j].float(), feat_b_pool[i][i].float())
                d_23 = self.l1(feat_a_fake_B[i][j].float(), feat_a_rec_B[i][j].float())
                c1 = d_24 + d_34 + 0.2 * d_23
                c2 = (d_21 + d_31 + 1e-7)

                contra = c1 / c2
                total_dis_loss += contra


        return total_dis_loss / len(self.adv_nce_layers)

    def calculate_DisNCE_loss2(self, src, tgt, tgt1, rtgt):

        # timestrap1 = time.time()
        feat_real_A = self.netD_B(src, self.adv_nce_layers, encode_only=True)
        feat_fake_A = self.netD_B(tgt, self.adv_nce_layers, encode_only=True)
        feat_rec_A = self.netD_B(tgt1, self.adv_nce_layers, encode_only=True)
        feat_real_B = self.netD_B(rtgt, self.adv_nce_layers, encode_only=True)

        feat_a_real_A = self.netAdvBackSample(feat_real_A)
        feat_a_fake_A = self.netAdvBackSample(feat_fake_A)
        feat_a_rec_A = self.netAdvBackSample(feat_rec_A)
        feat_b_pool = self.netAdvrainSample(feat_real_B)

        total_dis_loss = 0.0
        for i in range(len(feat_a_real_A)):
            for j in range(len(feat_a_real_A[i])):

                d_21 = self.l1(feat_a_fake_A[i][j].float(), feat_a_real_A[i][j].float())
                d_31 = self.l1(feat_a_rec_A[i][j].float(), feat_a_real_A[i][j].float())
                d_24 = self.l1(feat_a_fake_A[i][j].float(), feat_b_pool[i][j].float())
                d_34 = self.l1(feat_a_rec_A[i][j].float(), feat_b_pool[i][j].float())
                d_23 = self.l1(feat_a_fake_A[i][j].float(), feat_a_rec_A[i][j].float())
                c1 = d_21 + d_31 + 0.2 * d_23
                c2 = (d_24 + d_34 + 1e-7)
                contra = c1 / c2
                total_dis_loss += contra


        return total_dis_loss / len(self.adv_nce_layers)


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = vgg16(torch.load("vgg16-397923af.pth")).features.cuda()
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])

        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
       
        h = self.to_relu_1_2(x)
      
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h

        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3)
        return out

    def generate_visuals_for_evaluation(self, data, mode):
        with torch.no_grad():
            visuals = {}
            AtoB = self.opt.direction == "AtoB"
            G = self.netG_A
            source = data["A" if AtoB else "B"].to(self.device)
            if mode == "forward":
                visuals["fake_B"] = G(source)
            else:
                raise ValueError("mode %s is not recognized" % mode)
            return visuals


import torch.nn.functional as F
def norm(t):
    return F.normalize(t, dim=1, eps=1e-10)


def cosine_similarity(X, Y):
    '''
    compute cosine similarity for each pair of image
    Input shape: (batch,channel,H,W)
    Output shape: (batch,1)
    '''
    b, c, h, w = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
    X = X.reshape(b, c, h * w)
    Y = Y.reshape(b, c, h * w)
    corr = norm(X) * norm(Y)  # (B,C,H*W)
    similarity = corr.sum(dim=1).mean(dim=1)
    return similarity

class Blur(nn.Module):
        def __init__(self, nc):
            super(Blur, self).__init__()
            
            kernel = self.gauss_kernel(21,3,nc)

            kernel = torch.from_numpy(kernel).permute(3, 2, 0, 1)
            self.weight = nn.Parameter(data=kernel, requires_grad=False)

        def forward(self, x):
            x = F.conv2d(x, self.weight, stride=1, padding=10)
            return x

        def gauss_kernel(self,kernlen=21, nsig=3,channels=3):
            interval = (2 * nsig + 1.) / (kernlen)
            x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
            kern1d = np.diff(st.norm.cdf(x))
            kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
            kernel = kernel_raw / kernel_raw.sum()
            out_filter = np.array(kernel, dtype=np.float32)
            out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
            out_filter = np.repeat(out_filter, channels, axis=2)
            out_filter = np.repeat(out_filter, channels, axis=3)
            return out_filter