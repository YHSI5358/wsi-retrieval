from hipa_clip import get_hipa_clip_tokenizer, get_hipa_clip

tokenizer = get_hipa_clip_tokenizer()
model, transform = get_hipa_clip(checkpoint="./hipa_clip_ep15.pt", remove_text=False, remove_visual=False)
print(model)
exit()
# usage
model.to("cuda")
import torch
with torch.no_grad(), torch.cuda.amp.autocast():
    loaded_image = "something"
    loaded_image = transform(loaded_image).to(model.device)
    encoded_image = model.encode_image(loaded_image)[0]
    loaded_text = "something"
    loaded_text = tokenizer(loaded_text).to(model.device)
    encoded_text = model.encode_text(loaded_text)[0]
