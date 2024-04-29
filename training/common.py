import torch


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate_nodeclass(args, g, model, feature, labels, val_nid, test_nid):
    model = model.module
    model.eval()
    with torch.no_grad():
        if args.infer_method == "node":
            infer_seeds = torch.cat([val_nid, test_nid])
            pred, result_map = model.nodewise_inference(
                g, feature, infer_seeds, args.batch_size_eval)
            val_pred = pred[result_map[val_nid]]
            test_pred = pred[result_map[test_nid]]
        else:
            pred = model.layerwise_inference(g, feature, args.batch_size_eval)
            val_pred = pred[val_nid]
            test_pred = pred[test_nid]
    model.train()
    return compute_acc(val_pred, labels[val_nid].long()), compute_acc(
        test_pred, labels[test_nid].long())
