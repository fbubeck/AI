?($	/??<???<?D7???n4??@???!Έ?????$	܉??ko@???i?? @d?1?c@!U`MM?'@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$n4??@????c]?F??AQk?w????Y/?$???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?;Nё\??A?c?]K??A??<,Ԛ??Y/n????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?m4??@??Dio?????A=?U????Y????Mb??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Zd;?O???/?$???A?ǘ?????Y???H??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&:#J{?/?????T????A?HP???YL7?A`???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??	h"???????A?`TR'???Y?<,Ԛ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?镲q??|??Pk???A)\???(??Y??H?}??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ףp=
???)\???(??An4??@???Yvq?-??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???T????}?5^?I??A,e?X??Y???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	??MbX??}??b???A????Q??Y)\???(??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
??(?????1??%??A??H.???Y?ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&7?A`????mV}??b??Aףp=
???YU???N@??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&гY?????<?R?!???Aڬ?\m???Y??_vO??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&t$???~???*??	??A_?Q???Y????Mb??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Έ??????j+?????A?HP???Y?j+??ݣ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?uq???Ԛ?????AaTR'????Y;?O??n??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&jM?????Ǻ????A|a2U0??Y??\m????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??@?????x??#????Aꕲq???Y???B?i??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&,e?X???-?????A?Pk?w???Y????????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&;pΈ??????ͪ????A?0?*??YB>?٬???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???????\?C????A??Q???Y?U???؟?*	?????)?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatU0*????!?PT??1D@)~8gDi??1?{??u?B@:Preprocessing2F
Iterator::Model?T???N??!? ????@)???ZӼ??1?N???&3@:Preprocessing2U
Iterator::Model::ParallelMapV2?C??????!C%??(@)?C??????1C%??(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip"??u????!?÷??Q@)M?O???1????$@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??7??d??!,ޑɛ)@)9??v????1?u???@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??y?)??!_?|2?b@)??y?)??1_?|2?b@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap]m???{??!???q??1@)0L?
F%??1Z???B?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*w-!?l??!QT??1?@)w-!?l??1QT??1?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 47.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?^+?Օ@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	K??ھ???/?S????c]?F??!?*??	??	!       "	!       *	!       2$	{?????%cY???????Q??!?HP???:	!       B	!       J$	q	l9?*???Ҳ@|????ZӼ???!/?$???R	!       Z$	q	l9?*???Ҳ@|????ZӼ???!/?$???JCPU_ONLYY?^+?Օ@b Y      Y@q;x`Z??B@"?
both?Your program is POTENTIALLY input-bound because 47.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?37.9294% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 