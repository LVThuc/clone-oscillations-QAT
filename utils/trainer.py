import wandb
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import Checkpoint, global_step_from_engine


def create_trainer_engine(
	model,
	optimizer,
	criterion,
	metrics,
	data_loaders,
	lr_scheduler=None,
	save_checkpoint_dir=None,
	device='cuda',
	project_name='clone-oscillations-QAT',
):
	# Init W&B
	wandb.init(project=project_name)
	wandb.watch(model, log='all', log_freq=100)

	# Create trainer
	trainer = create_supervised_trainer(
		model=model,
		optimizer=optimizer,
		loss_fn=criterion,
		device=device,
		output_transform=custom_output_transform,
	)

	for name, metric in metrics.items():
		metric.attach(trainer, name)

	# Add lr_scheduler
	if lr_scheduler:
		trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: lr_scheduler.step())

	# Create evaluator
	evaluator = create_supervised_evaluator(model=model, metrics=metrics, device=device)

	# Save model checkpoint
	if save_checkpoint_dir:
		to_save = {'model': model, 'optimizer': optimizer}
		if lr_scheduler:
			to_save['lr_scheduler'] = lr_scheduler
		checkpoint = Checkpoint(
			to_save,
			save_checkpoint_dir,
			n_saved=1,
			global_step_transform=global_step_from_engine(trainer),
		)
		trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint)

	# Add hooks for logging metrics
	trainer.add_event_handler(Events.EPOCH_COMPLETED, log_training_results, optimizer)
	trainer.add_event_handler(
		Events.EPOCH_COMPLETED, run_evaluation_for_training, evaluator, data_loaders.val_loader
	)

	return trainer, evaluator


def custom_output_transform(x, y, y_pred, loss):
	return y_pred, y


def log_training_results(trainer, optimizer):
	learning_rate = optimizer.param_groups[0]['lr']
	log_metrics(trainer.state.metrics, 'Training', trainer.state.epoch, learning_rate)


def run_evaluation_for_training(trainer, evaluator, val_loader):
	evaluator.run(val_loader)
	log_metrics(evaluator.state.metrics, 'Evaluation', trainer.state.epoch)


def log_metrics(metrics, stage: str = '', training_epoch=None, learning_rate=None):
	log_text = '  {}'.format(metrics) if metrics else ''
	if training_epoch is not None:
		log_text = 'Epoch: {}'.format(training_epoch) + log_text
	if learning_rate and learning_rate > 0.0:
		log_text += '  Learning rate: {:.2E}'.format(learning_rate)
	log_text = 'Results - ' + log_text
	if stage:
		log_text = '{} '.format(stage) + log_text
	print(log_text, flush=True)

	# --- thêm wandb log ---
	log_dict = {}
	if training_epoch is not None:
		log_dict['epoch'] = training_epoch
	if learning_rate is not None:
		log_dict['lr'] = learning_rate
	for k, v in metrics.items():
		log_dict[f'{stage}/{k}'] = v
	wandb.log(log_dict)
