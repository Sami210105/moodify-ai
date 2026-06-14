import Navbar from './components/Navbar'
import Hero from './components/Hero'
import MarqueeTicker from './components/MarqueeTicker'
import MoodsSection from './components/MoodsSection'
import Detector from './components/Detector'
import AboutFooter from './components/AboutFooter'

const TICKER_ITEMS = [
  'Happy vibes', 'Sad songs', 'Angry anthems', 'Calm waves',
  'Romantic evenings', 'Anxious minds', 'GoEmotions AI', 'Last.fm powered',
  'Feel it deeply', 'Pure imagination', 'Music for every soul',
]

const TICKER2 = [
  'Feel it', 'Find it', 'Play it',
  'AI-powered', 'Last.fm', 'Real emotions',
  'Any genre', 'Any hour', 'Any mood',
]

export default function App() {
  return (
    <div style={{ overflowX:'hidden' }}>
      <Navbar />
      <Hero />      
      <MoodsSection />
      <MarqueeTicker items={TICKER_ITEMS} bg="#FFD93D" color="#0d0b14" />
      <Detector />
      <AboutFooter />
    </div>
  )
}
