?($	B6??H=??,0??????O??n??!]?Fx??$	?C????@?_??D@?<`J?,@!??Cws(@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?V-????<,Ԛ??A?0?*??Y??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?n????????????A?HP???YjM??S??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&]?Fx???sF????AM?J???Y*??Dؠ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??St$????x?&1??A?X????Y?&S???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&O??e???u?V??A?"??~j??YV}??b??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?+e?X??B`??"???A&䃞ͪ??Y46<???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&z6?>W???x?&1??AD?l?????Y? ?	???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&`vOj????	h"??A3ı.n???YL7?A`???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???_vO????v????A<Nё\???Y?J?4??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?z?G????rh??|??AB>?٬???Y??ǘ????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
??d?`T??\ A?c???Aj?t???Y9??v????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????????0?'???A?/?'??YU???N@??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&M?O???M?O????A?=yX???Y??H?}??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&+??	h??!?lV}??AK?46??Y?g??s???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&EGr?????:M???Ao??ʡ??YJ+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???????,C????A_?L???YM?O???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&x$(~??`vOj??A??ZӼ???Y/n????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&X9??v???ͪ??V??AK?=?U??Yn????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&0*??D???D???J??A??y?)??YX?5?;N??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&X?2ı.?????Q???A_?Q???Yy?&1???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?O??n????H.?!??A???&??Ye?X???*	    ??@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??H.???!7??ϩA@)=?U?????1m?큍^@@:Preprocessing2F
Iterator::Model??B?i???!һ"???A@)??u????1?4q??4@:Preprocessing2U
Iterator::Model::ParallelMapV2-!?lV??!ʅ??A?,@)-!?lV??1ʅ??A?,@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipaTR'????!?n-;P@)?f??j+??1??TR@?%@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate??HP??!?J
H??+@)!?rh????1?6zWd@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?\m?????!????/@)?\m?????1????/@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?O??n??!'c???2@)a2U0*???1`?wE?K@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?0?*??!?\???@)?0?*??1?\???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 48.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9]?|?\@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	8?8????񽫍????H.?!??!?sF????	!       "	!       *	!       2$	_hŋc??[jϔnE?????&??!M?J???:	!       B	!       J$	B??B?c???R?8
'????H?}??!??????R	!       Z$	B??B?c???R?8
'????H?}??!??????JCPU_ONLYY]?|?\@b Y      Y@q??????A@"?
both?Your program is POTENTIALLY input-bound because 48.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?35.4136% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 