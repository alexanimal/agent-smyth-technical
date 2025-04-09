import avatar from "../assets/avatar.png"

export default function Avatar(){
    
    return (
        <div className="avatar px-4">
            <img src={avatar} alt="avatar" className="rounded-full w-12 h-12" />
        </div>
    )
}