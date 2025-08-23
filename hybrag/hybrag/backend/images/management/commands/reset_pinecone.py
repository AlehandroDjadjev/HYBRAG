from __future__ import annotations
from django.core.management.base import BaseCommand
from django.conf import settings
from images.views import get_siglip
from pinecone import Pinecone

class Command(BaseCommand):
	help = 'Drop and recreate the Pinecone index to match current embedding dimension.'

	def add_arguments(self, parser):
		parser.add_argument('--dimension', type=int, default=0, help='Override dimension (otherwise inferred)')
		parser.add_argument('--metric', type=str, default='cosine', help='Similarity metric')
		parser.add_argument('--delete_only', action='store_true', help='Only delete the index (do not recreate)')

	def handle(self, *args, **options):
		pc = Pinecone(api_key=settings.PINECONE_API_KEY)
		index_name = settings.PINECONE_INDEX

		# infer dimension from current model if not provided
		dim = options['dimension'] or (get_siglip().dim or 512)
		metric = options['metric']

		self.stdout.write(self.style.NOTICE(f'Deleting index {index_name} if it exists...'))
		try:
			pc.delete_index(index_name)
		except Exception:
			pass

		if options['delete_only']:
			self.stdout.write(self.style.SUCCESS('Index deleted (if existed).'))
			return

		self.stdout.write(self.style.NOTICE(f'Creating index {index_name} dim={dim} metric={metric}'))
		pc.create_index(
			name=index_name,
			dimension=dim,
			metric=metric,
			spec={
				"serverless": {
					"cloud": getattr(settings, 'PINECONE_CLOUD', 'aws'),
					"region": getattr(settings, 'PINECONE_REGION', 'us-east-1')
				}
			}
		)
		self.stdout.write(self.style.SUCCESS('Index created. Update PINECONE_HOST to the new host URL from console.'))
