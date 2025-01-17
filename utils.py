from discord_webhook import DiscordWebhook, DiscordEmbed


def send_discord_file(file_path, webhook_url, title, description):
    webhook = DiscordWebhook(url=webhook_url)

    gen_file_name = file_path.name
    with open(file_path, "rb") as f:
        webhook.add_file(file=f.read(), filename=gen_file_name)

    embed = DiscordEmbed(
        title=title,
        description=description,
        color="03b2f8",
    )
    embed.set_thumbnail(url="attachment://" + gen_file_name)
    webhook.add_embed(embed)
    webhook.execute()
