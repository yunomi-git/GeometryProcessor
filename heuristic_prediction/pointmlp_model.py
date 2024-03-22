import pointmlp.classification_ModelNet40.models.pointmlp as pointmlp

def defaultPointMLPModel(args, **kwargs) -> pointmlp.PointMLP:
    return pointmlp.PointMLP(points=args["num_points"], class_num=args["num_outputs"], embed_dim=args["emb_dims"],
                             groups=args["groups"], res_expansion=args["res_expansion"],
                             activation=args["activation"], bias=args["bias"], use_xyz=args["use_xyz"], normalize=args["normalize"],
                             dim_expansion=args["dim_expansion"], pre_blocks=args["pre_blocks"], pos_blocks=args["pos_blocks"],
                             k_neighbors=args["k_neighbors"], reducers=args["reducers"], **kwargs)