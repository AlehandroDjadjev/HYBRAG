import { useEffect, useState } from 'react'

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://127.0.0.1:8000'

type Result = {
	id: string
	score: number
	image_url: string
	building: string
	shot_date: string
	notes?: string
}

export default function SearchPage() {
	const [q, setQ] = useState('')
	const [building, setBuilding] = useState('')
	const [from, setFrom] = useState('')
	const [to, setTo] = useState('')
	const [k, setK] = useState(5)
	const [results, setResults] = useState<Result[]>([])
	const [loading, setLoading] = useState(false)

	const runSearch = async () => {
		const params = new URLSearchParams()
		if (q) params.set('q', q)
		if (building) params.set('building', building)
		if (from) params.set('date_from', from)
		if (to) params.set('date_to', to)
		if (k) params.set('k', String(k))
		setLoading(true)
		try {
			const res = await fetch(`${API_BASE}/api/search?${params.toString()}`)
			const data = await res.json()
			setResults(data.results || [])
		} finally {
			setLoading(false)
		}
	}

	useEffect(() => {
		// no-op
	}, [])

	return (
		<div style={{ padding: 24, fontFamily: 'sans-serif' }}>
			<h2>Search Images</h2>
			<div style={{ display: 'grid', gap: 8, maxWidth: 560 }}>
				<input placeholder="Text query" value={q} onChange={(e) => setQ(e.target.value)} />
				<input placeholder="Building (optional)" value={building} onChange={(e) => setBuilding(e.target.value)} />
				<label>From <input type="date" value={from} onChange={(e) => setFrom(e.target.value)} /></label>
				<label>To <input type="date" value={to} onChange={(e) => setTo(e.target.value)} /></label>
				<label>Top K <input type="number" value={k} onChange={(e) => setK(Number(e.target.value))} /></label>
				<button disabled={loading} onClick={runSearch}>{loading ? 'Searching...' : 'Search'}</button>
			</div>
			<div style={{ marginTop: 16, display: 'grid', gap: 12 }}>
				{results.map((r) => (
					<div key={r.id} style={{ border: '1px solid #ddd', padding: 12 }}>
						<div><strong>Score:</strong> {r.score.toFixed(3)}</div>
						<div><img src={r.image_url} alt={r.id} style={{ maxWidth: 320 }} /></div>
						<div><strong>Building:</strong> {r.building}</div>
						<div><strong>Date:</strong> {r.shot_date}</div>
						{r.notes ? <div><strong>Notes:</strong> {r.notes}</div> : null}
					</div>
				))}
			</div>
		</div>
	)
}
