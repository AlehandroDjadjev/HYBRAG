import { useState } from 'react'

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://127.0.0.1:8000'

export default function UploadPage() {
	const [file, setFile] = useState<File | null>(null)
	const [building, setBuilding] = useState('')
	const [shotDate, setShotDate] = useState('')
	const [notes, setNotes] = useState('')
	const [resp, setResp] = useState<any>(null)
	const [loading, setLoading] = useState(false)

	const [files, setFiles] = useState<FileList | null>(null)
	const [bulkBuilding, setBulkBuilding] = useState('')
	const [bulkDate, setBulkDate] = useState('')
	const [bulkNotes, setBulkNotes] = useState('')
	const [bulkResp, setBulkResp] = useState<any>(null)
	const [bulkLoading, setBulkLoading] = useState(false)

	const onSubmit = async (e: React.FormEvent) => {
		e.preventDefault()
		if (!file || !building || !shotDate) return
		const fd = new FormData()
		fd.append('file', file)
		fd.append('building', building)
		fd.append('shot_date', shotDate)
		if (notes) fd.append('notes', notes)
		setLoading(true)
		try {
			const res = await fetch(`${API_BASE}/api/images`, { method: 'POST', body: fd })
			const data = await res.json()
			setResp(data)
		} finally {
			setLoading(false)
		}
	}

	const onBulkSubmit = async (e: React.FormEvent) => {
		e.preventDefault()
		if (!files || !bulkBuilding || !bulkDate) return
		const fd = new FormData()
		Array.from(files).forEach((f) => fd.append('files', f))
		// For simplicity, apply same building/date/notes to all files
		Array.from(files).forEach(() => fd.append('building', bulkBuilding))
		Array.from(files).forEach(() => fd.append('shot_date', bulkDate))
		Array.from(files).forEach(() => fd.append('notes', bulkNotes))
		setBulkLoading(true)
		try {
			const res = await fetch(`${API_BASE}/api/images/batch`, { method: 'POST', body: fd })
			const data = await res.json()
			setBulkResp(data)
		} finally {
			setBulkLoading(false)
		}
	}

	return (
		<div style={{ padding: 24, fontFamily: 'sans-serif' }}>
			<h2>Upload Image</h2>
			<form onSubmit={onSubmit}>
				<div>
					<input type="file" accept="image/*" onChange={(e) => setFile(e.target.files?.[0] || null)} />
				</div>
				<div>
					<input placeholder="Building" value={building} onChange={(e) => setBuilding(e.target.value)} />
				</div>
				<div>
					<input type="date" value={shotDate} onChange={(e) => setShotDate(e.target.value)} />
				</div>
				<div>
					<textarea placeholder="Notes (optional)" value={notes} onChange={(e) => setNotes(e.target.value)} />
				</div>
				<button disabled={loading} type="submit">{loading ? 'Uploading...' : 'Upload'}</button>
			</form>
			{resp && (
				<pre style={{ marginTop: 16 }}>{JSON.stringify(resp, null, 2)}</pre>
			)}

			<hr style={{ margin: '24px 0' }} />
			<h2>Bulk Upload</h2>
			<form onSubmit={onBulkSubmit}>
				<div>
					<input multiple type="file" accept="image/*" onChange={(e) => setFiles(e.target.files)} />
				</div>
				<div>
					<input placeholder="Building" value={bulkBuilding} onChange={(e) => setBulkBuilding(e.target.value)} />
				</div>
				<div>
					<input type="date" value={bulkDate} onChange={(e) => setBulkDate(e.target.value)} />
				</div>
				<div>
					<textarea placeholder="Notes (optional for all)" value={bulkNotes} onChange={(e) => setBulkNotes(e.target.value)} />
				</div>
				<button disabled={bulkLoading} type="submit">{bulkLoading ? 'Uploading...' : 'Bulk Upload'}</button>
			</form>
			{bulkResp && (
				<pre style={{ marginTop: 16 }}>{JSON.stringify(bulkResp, null, 2)}</pre>
			)}
		</div>
	)
}
