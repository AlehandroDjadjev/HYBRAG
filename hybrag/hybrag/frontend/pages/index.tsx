import Link from 'next/link'

export default function Home() {
	return (
		<div style={{ padding: 24, fontFamily: 'sans-serif' }}>
			<h1>HYBRAG Demo</h1>
			<ul>
				<li><Link href="/upload">Upload Image</Link></li>
				<li><Link href="/search">Search Images</Link></li>
			</ul>
		</div>
	)
}
