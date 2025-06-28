from sweep_config import *
from utils import *
from dependencies import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True,
            choices=['FacebookAI/roberta-base', 'facebook/opt-350m', 'GroNLP/hateBERT', 'openai-community/gpt2'])
    parser.add_argument('--strategy', type=str, required=True,
                       choices=['multiclass_scratch', 'one_v_rest', 'pair_wise'])
    parser.add_argument('--directory', type=str, required=True, default='role_all',
                       choices=['role_all', 'role_off'])
    parser.add_argument('--data_point', type=str, default='llm_both')
    parser.add_argument('--fold', type=str, default='1')
    parser.add_argument('--sweep_count', type=int, default=1)
    parser.add_argument('--up_sample', type=str, required=True,
            choices=['yes', 'no'], default='no')
    
    args = parser.parse_args()

    return args

def parse_data(train_df, val_df, test_df, gold_df):
    # Convert labels to numeric
    train_df.ROLE = train_df.ROLE.apply(map_roles)
    train_df.HATE = train_df.HATE.apply(map_binary)
    val_df.ROLE = val_df.ROLE.apply(map_roles)
    val_df.HATE = val_df.HATE.apply(map_binary)
    test_df.HATE = test_df.HATE.apply(map_binary)
    test_df.ROLE = test_df.ROLE.apply(map_roles)
    gold_df.ROLE = gold_df.ROLE.apply(map_roles)
    
    # Prepare data
    train_texts = train_df.TEXT.values.tolist()
    train_labels = train_df.ROLE.values.tolist()
    val_texts = val_df.TEXT.values.tolist()
    val_labels = val_df.ROLE.values.tolist()
    test_texts = test_df.TEXT.values.tolist()
    test_labels = test_df.ROLE.values.tolist()
    gold_texts = gold_df.TEXT.values.tolist()
    gold_labels = gold_df.ROLE.values.tolist()

    binary_train_labels = train_df.HATE.values.tolist()
    binary_val_labels = val_df.HATE.values.tolist()
    binary_test_labels = test_df.HATE.values.tolist()

    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, gold_texts, gold_labels, binary_train_labels, binary_val_labels, binary_test_labels

def parse_downstream_data(df_train, df_val, df_test, df_gold):
    for df in [df_train, df_val, df_test, df_gold]:
        df.ROLE = df.ROLE.apply(map_roles)

    train_texts = df_train.TEXT.tolist()
    val_texts = df_val.TEXT.tolist()
    test_texts = df_test.TEXT.tolist()
    gold_texts = df_gold.TEXT.tolist()

    train_labels = df_train.ROLE.tolist()
    val_labels = df_val.ROLE.tolist()
    test_labels = df_test.ROLE.tolist()
    gold_labels = df_gold.ROLE.tolist()

    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, gold_texts, gold_labels


def main():
    args = get_args()

    project_name = f"{args.model_name.split('/')[-1]}_{args.strategy}_{args.data_point}_{args.fold}"
    wandb.init(project=project_name)
    config = wandb.config

    # Create training config from wandb config
    training_config = TrainingConfig(
        model_name=args.model_name,
        batch_size=config.batch_size,
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        l2_lambda=config.l2_lambda,
        ewc_lambda=getattr(config, 'ewc_lambda', 1000.0),
        dropout_rate=config.dropout_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        output_dir=f"./output_{wandb.run.id}",
        use_focal_loss=config.use_focal_loss,
        focal_alpha=config.focal_alpha,
        focal_gamma=config.focal_gamma,
    )

    # Load data
    up_train, up_val, down_train, down_val, updown_test, updown_gold = get_en_data(args.directory, args.data_point, args.fold, args.strategy)

    if args.up_sample == 'yes':
        logger.info("Doing Upsampling")
        train_df = upsample_minority_classes(down_train, label_column='ROLE')
        print(Counter(train_df.ROLE))
    else:
        train_df = down_train.copy()

    val_df = down_val.copy()

    #down_train = down_train.head(10)
    #down_val = down_val.head(5)
    #down_test = down_test.head(2)

    updown_test = updown_test[['TEXT', 'HATE', 'ROLE']]
    test_df = updown_test.copy()
    gold_df = updown_gold.copy()
    
    #test_df = test_df.head(10)
    #gold_df = gold_df.head(5)

    # Initialize trainer
    trainer = BullyingClassifierTrainer(training_config)

    results = {}
    try:
        if args.strategy == 'multiclass_scratch':
            # Log data balance
            print_data_balance_report(train_df, val_df, test_df, 'ROLE', gold_df)

            train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, gold_texts, gold_labels = parse_downstream_data(train_df, val_df, test_df, gold_df)

            logger.info("Executing multiclass from scratch strategy")
            results = trainer.train_multiclass_from_scratch(
                train_texts, train_labels, val_texts, val_labels
            )

            # Log metrics to wandb
            wandb.log({
                'strategy': 'multiclass_scratch',
                'val_f1_macro': results['results']['final_metrics']['f1_macro'],
                'val_f1_weighted': results['results']['final_metrics']['f1_weighted'],
                'val_accuracy': results['results']['final_metrics']['accuracy'],
                'val_loss': results['results']['final_metrics']['avg_loss'],
                'best_f1': results['results']['best_f1']
            })

            model = results['model']

            test_df = trainer.predict_multiclass(model, test_texts, test_labels)
            gold_df = trainer.predict_multiclass(model, gold_texts, gold_labels)

            run_name = wandb.run.name
            project_name = wandb.run.project
            
            test_f1 = f1_score(test_df.true_label, test_df.predicted_label, average='macro', zero_division=0.0)
            gold_f1 = f1_score(gold_df.true_label, gold_df.predicted_label, average='macro', zero_division=0.0)

            mname = args.model_name.split("/")[-1]
            csv_name = f"{project_name}_{run_name}_{args.directory}_upsample={args.up_sample}"
            test_df.to_csv(f"{csv_name}_test_{round(test_f1, 2)}.csv", index=False)
            gold_df.to_csv(f"{csv_name}_goldTest_{round(gold_f1, 2)}.csv", index=False)

            wandb.log({
                'test_f1_macro': test_f1,
                'gold_f1_macro': gold_f1,
            })

        elif args.strategy == 'pair_wise':

            logger.info("Executing hierarchical role-pair classifiers strategy")
            print()

            print_data_balance_report(train_df, val_df, test_df, 'ROLE', gold_df)

            train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, gold_texts, gold_labels = parse_downstream_data(train_df, val_df, test_df, gold_df)

            results = trainer.train_hierarchical_pairwise_classifiers(train_texts, train_labels, val_texts, val_labels)

            pair_wise_preds = trainer.predict_hierarchical_pairwise(results['models'], test_texts, test_labels, uncertainty_threshold=0.1)

            pair_wise_gold = trainer.predict_hierarchical_pairwise(results['models'], gold_texts, gold_labels, uncertainty_threshold=0.1)

            
            ## Logging Each Epoch Metrics
            for label, data in results.items():
                history = data.get('history', [])
                for epoch_data in history:
                    wandb.log({
                        f"{label}/epoch": epoch_data['epoch'],
                        f"{label}/train_loss": epoch_data['train_loss'],
                        f"{label}/val_f1": epoch_data['val_f1'],
                        f"{label}/val_accuracy": epoch_data['val_accuracy'],
                    }, step=epoch_data['epoch'])

            for label, data in results.items():
                metrics = data.get('final_metrics', {})
                wandb.log({
                    f"{label}/final_accuracy": metrics.get('accuracy', 0.0),
                    f"{label}/final_f1_weighted": metrics.get('f1_weighted', 0.0),
                    f"{label}/final_f1_macro": metrics.get('f1_macro', 0.0),
                    f"{label}/final_avg_loss": metrics.get('avg_loss', 0.0),
                })

            run_name = wandb.run.name
            project_name = wandb.run.project
            test_f1 = f1_score(pair_wise_preds.true_label, pair_wise_preds.predicted_label, average='macro', zero_division=0.0)
            gold_f1 = f1_score(pair_wise_gold.true_label, pair_wise_gold.predicted_label, average='macro', zero_division=0.0)
            mname = args.model_name.split("/")[-1]
            csv_name = f"{project_name}_{run_name}_{args.directory}"
            pair_wise_preds.to_csv(f"{csv_name}_test_{round(test_f1, 2)}.csv", index=False)
            pair_wise_gold.to_csv(f"{csv_name}_goldTest_{round(gold_f1, 2)}.csv", index=False)

            wandb.log({
                'test_f1_macro': test_f1,
                'gold_f1_macro': gold_f1,
            })

            test_report = classification_report(pair_wise_preds.true_label, pair_wise_preds.predicted_label)
            gold_report = classification_report(pair_wise_gold.true_label, pair_wise_gold.predicted_label)

            # Log to wandb
            wandb.log({
                'test_classification_report': wandb.Html(f"<pre>{test_report}</pre>"),
                'gold_classification_report': wandb.Html(f"<pre>{gold_report}</pre>"),
            })

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        wandb.log({'error': str(e), 'status': 'failed'})
        raise e

    finally:
        # Clean up GPU memory
        torch.cuda.empty_cache()
        gc.collect()

    logger.info(f"Completed {args.strategy} strategy")

if __name__ == "__main__":
    args = get_args()
    project_name = f"{args.model_name.split('/')[-1]}_{args.strategy}_{args.data_point}_{args.fold}_upsample={args.up_sample}" 
    sweep_count = args.sweep_count
    sweep_config_info = get_base_sweep_config()
    sweep_id = wandb.sweep(sweep_config_info, project=project_name)

    wandb.agent(sweep_id, main, count=sweep_count)