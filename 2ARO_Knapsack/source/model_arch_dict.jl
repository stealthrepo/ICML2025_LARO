using DataStructures

function instance_arch_10()
    ordered_model_architecture = OrderedDict{String, Any}()
    ordered_model_architecture["input_layers"] = ["instance", "X", "U"]

    ordered_model_architecture["hidden_layers"] = OrderedDict{String, Any}()

    ordered_model_architecture["hidden_layers"]["embedding_instance"] = [ordered_model_architecture["input_layers"][1]]
    ordered_model_architecture["hidden_layers"]["embedding_X"] = [ordered_model_architecture["input_layers"][2]]
    ordered_model_architecture["hidden_layers"]["embedding_uncern"] = [ordered_model_architecture["input_layers"][3]]

    ordered_model_architecture["hidden_layers"]["fc1"] = ["embedding_instance", "embedding_X", "embedding_uncern"]
    ordered_model_architecture["hidden_layers"]["fc2"] = ["fc1"]

    ordered_model_architecture["output_layers"] = OrderedDict{String, Any}()
    ordered_model_architecture["output_layers"]["fc3"] = ["fc2"]

    return ordered_model_architecture
end

function instance_arch()
    ordered_model_architecture = OrderedDict{String, Any}()
    ordered_model_architecture["input_layers"] = ["instance", "X", "U"]

    ordered_model_architecture["hidden_layers"] = OrderedDict{String, Any}()

    ordered_model_architecture["hidden_layers"]["embedding_instance"] = [ordered_model_architecture["input_layers"][1]]
    ordered_model_architecture["hidden_layers"]["embedding_X"] = [ordered_model_architecture["input_layers"][2]]
    ordered_model_architecture["hidden_layers"]["embedding_uncern"] = [ordered_model_architecture["input_layers"][3]]

    ordered_model_architecture["hidden_layers"]["fc1"] = ["embedding_instance", "embedding_X", "embedding_uncern"]
    ordered_model_architecture["hidden_layers"]["fc2"] = ["fc1"]
    ordered_model_architecture["hidden_layers"]["fc3"] = ["fc2"]
    #ordered_model_architecture["hidden_layers"]["fc4"] = ["fc3"]

    ordered_model_architecture["output_layers"] = OrderedDict{String, Any}()
    ordered_model_architecture["output_layers"]["fc4"] = ["fc3"]

    return ordered_model_architecture
end

true