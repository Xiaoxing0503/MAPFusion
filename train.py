# Training Unet network
# auto-encoder

import os
import sys
import time
import numpy as np
from tqdm import tqdm, trange
import scipy.io as scio
import random
import torch
from torch.optim import Adam
from torch.autograd import Variable
import utils
from net import Fuse_Unet
from args_fusion import args
import pytorch_msssim
import loss
import loss_per1
import RGB_YCrCb

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main():

	original_ir_imgs_path = utils.list_images(args.ir_datasets)
	original_vi_imgs_path = utils.list_images(args.vi_datasets)
	train_num = 1024
	original_ir_imgs_path = original_ir_imgs_path[:train_num]
	original_vi_imgs_path = original_vi_imgs_path[:train_num]
	random.shuffle(original_ir_imgs_path)
	random.shuffle(original_vi_imgs_path)    
	# for i in range(5):
	i = 2
	train(i, original_ir_imgs_path, original_vi_imgs_path)


def train(i, original_ir_imgs_path, original_vi_imgs_path):

	batch_size = args.batch_size

	# load network model, RGB
	in_c = 3 # 1 - gray; 3 - RGB
	if in_c == 1:
		img_model = 'L'
	else:
		img_model = 'RGB'
	input_nc = 1
	output_nc = 1
	densefuse_model = Fuse_Unet(input_nc, output_nc)

	if args.resume is not None:
		print('Resuming, initializing using weight from {}.'.format(args.resume))
		densefuse_model.load_state_dict(torch.load(args.resume))
	print(densefuse_model)
	densefuse_model.cuda()
	optimizer = Adam(densefuse_model.parameters(), args.lr)
	# mse_loss = torch.nn.MSELoss()
	ssim_loss = pytorch_msssim.msssim

	tbar = trange(args.epochs)
	print('Start training.....')

	# creating save path
	temp_path_model = os.path.join(args.save_model_dir, args.ssim_path[i])
	if os.path.exists(temp_path_model) is False:
		os.mkdir(temp_path_model)

	temp_path_loss = os.path.join(args.save_loss_dir, args.ssim_path[i])
	if os.path.exists(temp_path_loss) is False:
		os.mkdir(temp_path_loss)

	Loss_pixel = []
	Loss_ssim = []
	Loss_per = []
	Loss_grad = []
	Loss_all = []
	all_ssim_loss = 0.
	all_pixel_loss = 0.
	all_grad_loss = 0.
	all_per_loss = 0.
	for e in tbar:
		print('Epoch %d.....' % e)
		# load training database
		image_set_ir, batches = utils.load_dataset(original_ir_imgs_path, batch_size)
		image_set_vi, batches = utils.load_dataset(original_vi_imgs_path, batch_size)
		densefuse_model.train()
		count = 0
		for batch in range(batches):
			image_paths_ir = image_set_ir[batch * batch_size:(batch * batch_size + batch_size)]
			image_paths_vi = image_set_vi[batch * batch_size:(batch * batch_size + batch_size)]
			img_ir = utils.get_train_images_auto(image_paths_ir, height=args.HEIGHT, width=args.WIDTH, mode='L')
			img_vi = utils.get_train_images_auto(image_paths_vi, height=args.HEIGHT, width=args.WIDTH, mode='RGB')
			img_ir = img_ir.cuda()
			img_vi = img_vi.cuda()
			count += 1
			optimizer.zero_grad()
			img_ir = Variable(img_ir, requires_grad=False)
			img_vi = Variable(img_vi, requires_grad=False)
			img_vi_ycrcb = RGB_YCrCb.RGB2YCrCb(img_vi)
			img_ir = img_ir.cuda()
			img_vi_ycrcb = img_vi_ycrcb.cuda()

			# get fusion image
			# encoder
			# e = densefuse_model.encoder1(img_ir,img_vi)
			e1_1,e2_1,e3_1,e4_1 = densefuse_model.encoder1(img_ir)
			e1_2,e2_2,e3_2,e4_2 = densefuse_model.encoder2(img_vi_ycrcb)
			# en = densefuse_model.fusion(en_ir,en_vi)
			# decoder
			outputs = densefuse_model.decoder(e1_1,e2_1,e3_1,e4_1,e1_2,e2_2,e3_2,e4_2)
			# resolution loss
			x_ir = Variable(img_ir.data.clone(), requires_grad=False)
			img_vi_ycrcb = Variable(img_vi_ycrcb.data.clone(), requires_grad=False)
			img_vi_ycrcb = img_vi_ycrcb[:,:1]

			ssim_loss_value = 0.
			pixel_loss_value = 0.
			per_loss_value = 0.
			grad_loss_value = 0.
			for output in outputs:
				pixel_loss_temp = loss.L_Int(x_ir, img_vi_ycrcb, output)
				ssim_ir_loss_temp = ssim_loss(output, x_ir, normalize=True)
				ssim_vi_loss_temp = ssim_loss(output, img_vi_ycrcb, normalize=True)
				per_loss_value_temp1,s1 = loss_per1.per_loss(x_ir, output)
				per_loss_value_temp2,s2 = loss_per1.per_loss(img_vi_ycrcb, output)
				w1 = s1 / (s1 + s2)
				w2 = s2 / (s1 + s2)
				grad_loss_value_temp1 = loss.L_Grad(x_ir, img_vi_ycrcb, output)


				ssim_loss_value += 0.5*(1-ssim_ir_loss_temp)+0.5*(1-ssim_vi_loss_temp)
				# ssim_loss_value += w1 * (1 - ssim_ir_loss_temp) + w2 * (1 - ssim_vi_loss_temp)
				pixel_loss_value += pixel_loss_temp
				# per_loss_value += 0.5*per_loss_value_temp1 + 0.5*per_loss_value_temp2
				per_loss_value += w1 * per_loss_value_temp1 + w2 * per_loss_value_temp2
				grad_loss_value += grad_loss_value_temp1

			ssim_loss_value /= len(outputs)
			pixel_loss_value /= len(outputs)
			per_loss_value /= len(outputs)
			grad_loss_value /= len(outputs)
			# total loss
			# total_loss = pixel_loss_value + grad_loss_value
			# total_loss = ssim_loss_value * 100 + 0.5 * pixel_loss_value + grad_loss_value
			total_loss =0.5 * pixel_loss_value + ssim_loss_value * 100 + grad_loss_value + 0.001 * per_loss_value
			total_loss.backward()
			optimizer.step()

			all_ssim_loss += ssim_loss_value.item()
			all_per_loss += per_loss_value.item()
			all_pixel_loss += pixel_loss_value.item()
			all_grad_loss += grad_loss_value.item()
			if (batch + 1) % args.log_interval == 0:
				mesg = "{}\tEpoch {}:\t[{}/{}]\t pixel loss: {:.6f}\t ssim loss: {:.6f}\t grad loss: {:.6f}\t per loss: {:.6f}\t total: {:.6f}".format(
					time.ctime(), e + 1, count, batches,
								  all_pixel_loss / args.log_interval,
								  all_ssim_loss / args.log_interval,
					              all_grad_loss / args.log_interval,
								  all_per_loss / args.log_interval,
								  (all_pixel_loss*0.5 + all_grad_loss + all_ssim_loss * 100+ all_per_loss*0.001) / args.log_interval
				)
				print(mesg)
				# tbar.set_description(mesg)
				Loss_pixel.append(all_pixel_loss / args.log_interval)
				Loss_ssim.append(all_ssim_loss / args.log_interval)
				Loss_per.append(all_per_loss / args.log_interval)
				Loss_grad.append(all_grad_loss / args.log_interval)
				# Loss_all.append((all_ssim_loss * 1000 + 50 * all_grad_loss + all_per_loss) / args.log_interval)
				Loss_all.append((all_pixel_loss*0.5 + all_grad_loss + all_ssim_loss * 100 + all_per_loss*0.001) / args.log_interval)
				all_ssim_loss = 0.
				all_pixel_loss = 0.
				all_per_loss = 0.
				all_grad_loss = 0.

			if (batch + 1) % (200 * args.log_interval) == 0:
				# save model
				densefuse_model.eval()
				# densefuse_model.cpu()
				densefuse_model.cuda()
				save_model_filename = args.ssim_path[i] + '/' + "Epoch_" + str(e) + "_iters_" + str(count) + "_" + \
									  str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[
										  i] + ".model"
				save_model_path = os.path.join(args.save_model_dir, save_model_filename)
				torch.save(densefuse_model.state_dict(), save_model_path)
				# # save loss data
				# pixel loss
				loss_data_pixel = np.array(Loss_pixel)
				loss_filename_path = args.ssim_path[i] + '/' + "loss_pixel_epoch_" + str(
					args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
									 args.ssim_path[i] + ".mat"
				save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
				scio.savemat(save_loss_path, {'loss_pixel': loss_data_pixel})
				# SSIM loss
				loss_data_ssim = np.array(Loss_ssim)
				loss_filename_path = args.ssim_path[i] + '/' + "loss_ssim_epoch_" + str(
					args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
									 args.ssim_path[i] + ".mat"
				save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
				scio.savemat(save_loss_path, {'loss_ssim': loss_data_ssim})
				# per loss
				loss_data_per = np.array(Loss_per)
				loss_filename_path = args.ssim_path[i] + '/' + "loss_per_epoch_" + str(
					args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
									 args.ssim_path[i] + ".mat"
				save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
				scio.savemat(save_loss_path, {'loss_per': loss_data_per})
				# grad loss
				loss_data_grad = np.array(Loss_grad)
				loss_filename_path = args.ssim_path[i] + '/' + "loss_grad_epoch_" + str(
					args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
									 args.ssim_path[i] + ".mat"
				save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
				scio.savemat(save_loss_path, {'loss_grad': loss_data_grad})
				# all loss
				loss_data_total = np.array(Loss_all)
				loss_filename_path = args.ssim_path[i] + '/' + "loss_total_epoch_" + str(
					args.epochs) + "_iters_" + str(count) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
									 args.ssim_path[i] + ".mat"
				save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
				scio.savemat(save_loss_path, {'loss_total': loss_data_total})

				densefuse_model.train()
				densefuse_model.cuda()
				tbar.set_description("\nCheckpoint, trained model saved at", save_model_path)

	# pixel loss
	loss_data_pixel = np.array(Loss_pixel)
	loss_filename_path = args.ssim_path[i] + '/' + "Final_loss_pixel_epoch_" + str(
		args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':','_') + "_" + \
						 args.ssim_path[i] + ".mat"
	save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
	scio.savemat(save_loss_path, {'loss_pixel': loss_data_pixel})
	# SSIM loss
	loss_data_ssim = np.array(Loss_ssim)
	loss_filename_path = args.ssim_path[i] + '/' + "Final_loss_ssim_epoch_" + str(
		args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
						 args.ssim_path[i] + ".mat"
	save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
	scio.savemat(save_loss_path, {'loss_ssim': loss_data_ssim})
	# grad loss
	loss_data_grad = np.array(Loss_grad)
	loss_filename_path = args.ssim_path[i] + '/' + "Final_loss_grad_epoch_" + str(
		args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
						 args.ssim_path[i] + ".mat"
	save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
	scio.savemat(save_loss_path, {'loss_grad': loss_data_grad})
	# per loss
	loss_data_per = np.array(Loss_per)
	loss_filename_path = args.ssim_path[i] + '/' + "Final_loss_per_epoch_" + str(
		args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
						 args.ssim_path[i] + ".mat"
	save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
	scio.savemat(save_loss_path, {'loss_per': loss_data_per})
	# all loss
	loss_data_total = np.array(Loss_all)
	loss_filename_path = args.ssim_path[i] + '/' + "Final_loss_total_epoch_" + str(
		args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + \
						 args.ssim_path[i] + ".mat"
	save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
	scio.savemat(save_loss_path, {'loss_total': loss_data_total})
	# save model
	densefuse_model.eval()
	# densefuse_model.cpu()
	densefuse_model.cuda()
	save_model_filename = args.ssim_path[i] + '/' "Final_epoch_" + str(args.epochs) + "_" + \
						  str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" + args.ssim_path[i] + ".model"
	save_model_path = os.path.join(args.save_model_dir, save_model_filename)
	torch.save(densefuse_model.state_dict(), save_model_path)

	print("\nDone, trained model saved at", save_model_path)


if __name__ == "__main__":
	main()
