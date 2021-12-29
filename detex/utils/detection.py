


def collect_detections(dets, score_threshold=0.5):

    preds = []

    for det in dets:
        raw_boxes = det["raw_boxes"]
        box_ids = det["box_ids"]
        boxes = det["boxes"]
        labels = det["labels"]
        scores = det["scores"]


        score_ids = (scores > score_threshold).nonzero().squeeze(1)
        ids = score_ids

        boxes, labels, scores, box_ids = boxes[ids], labels[ids], scores[ids], box_ids[ids]


        pred = {
            "raw_boxes": raw_boxes.detach().cpu().numpy(),
            "boxes": boxes.detach().cpu().numpy(), 
            "labels": labels.detach().cpu().numpy(), 
            "scores": scores.detach().cpu().numpy(),
            "box_ids": box_ids.detach().cpu().numpy()
        }
        preds.append(pred)

    return preds
